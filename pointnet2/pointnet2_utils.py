# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext_src as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)

import torch
from torch import Tensor

import torch

def furthest_point_sampling_py(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    r"""
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance.

    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint, representing the input point cloud.
    npoint : int
        Number of points to sample.

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the indices of the sampled points.
    """
    device = xyz.device
    B, N, _ = xyz.shape

    # Initialize the sampled indices tensor
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # Initialize the distance to the closest sampled point
    distance = torch.ones(B, N, device=device) * 1e10

    # Initialize the batch indices
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    # Randomly select the first point
    furthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = furthest

        # Get the coordinates of the current centroids
        centroid = xyz[batch_indices, furthest, :].view(B, 1, 3)

        # Compute the distance between the centroids and all other points
        dist = torch.sum((xyz - centroid) ** 2, dim=2)

        # Update the distance tensor to keep track of the minimum distance
        mask = dist < distance
        distance[mask] = dist[mask]

        # Select the point with the largest distance as the next point
        furthest = torch.max(distance, dim=1)[1]

    return centroids

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return furthest_point_sampling_py(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    r"""
    Gather points from the input features based on the given indices.

    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of input features.
    idx : torch.Tensor
        (B, npoint) tensor of indices to gather from the features.

    Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor of gathered features.
    """
    B, C = features.size(0), features.size(1)
    npoint = idx.size(1)

    # 扩展 idx 到 (B, C, npoint) 以匹配 features 的维度
    idx_expanded = idx.unsqueeze(1).expand(B, C, npoint)

    # 使用 torch.gather 提取对应点的特征
    gathered_features = torch.gather(features, 2, idx_expanded)

    return gathered_features

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


def three_nn_py(unknown, known):
    """
    Find the three nearest neighbors of each point in `unknown` from `known`.

    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of query points (points for which we find neighbors).
    known : torch.Tensor
        (B, m, 3) tensor of reference points (points to search neighbors from).

    Returns
    -------
    dist : torch.Tensor
        (B, n, 3) tensor containing the squared L2 distances to the three nearest neighbors.
    idx : torch.Tensor
        (B, n, 3) tensor containing the indices of the three nearest neighbors.
    """
    B, n, _ = unknown.shape
    _, m, _ = known.shape

    # Compute squared Euclidean distance between each unknown point and all known points
    dist = torch.cdist(unknown, known, p=2)  # Shape: (B, n, m)

    # Find the indices of the three nearest neighbors
    dist_sorted, idx_sorted = torch.topk(dist, k=3, dim=-1, largest=False, sorted=True)  # (B, n, 3)

    return dist_sorted, idx_sorted

class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = three_nn_py(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply

def three_interpolate_py(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    r"""
    Perform weighted linear interpolation of 3 nearest neighbors.

    Parameters
    ----------
    features : torch.Tensor
        (B, c, m) tensor of features to interpolate from.
    idx : torch.Tensor
        (B, n, 3) tensor of indices of the three nearest neighbors.
    weight : torch.Tensor
        (B, n, 3) tensor of weights for interpolation.

    Returns
    -------
    torch.Tensor
        (B, c, n) tensor of interpolated features.
    """
    B, c, m = features.size()
    n = idx.size(1)

    # Initialize the output tensor
    interpolated_features = torch.zeros((B, c, n), dtype=features.dtype, device=features.device)

    # Perform interpolation
    for b in range(B):
        for i in range(n):
            # Get the indices of the three nearest neighbors
            neighbor_idx = idx[b, i]  # (3,)
            # Get the weights for the three nearest neighbors
            neighbor_weight = weight[b, i]  # (3,)
            # Gather the features of the three nearest neighbors
            neighbor_features = features[b, :, neighbor_idx]  # (c, 3)
            # Perform weighted interpolation
            interpolated_features[b, :, i] = torch.sum(neighbor_features * neighbor_weight, dim=1)  # (c,)

    return interpolated_features

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return three_interpolate_py(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def group_points_py(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    r"""
    Group features according to the indices.

    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group.
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indices of features to group with.

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor of grouped features.
    """
    B, C, N = features.size()
    _, npoint, nsample = idx.size()

    # Initialize the output tensor
    grouped_features = torch.zeros((B, C, npoint, nsample), dtype=features.dtype, device=features.device)

    # Group the features
    for b in range(B):
        for i in range(npoint):
            for k in range(nsample):
                # Get the index of the feature
                index = idx[b, i, k]
                if index < N:  # Ensure the index is valid
                    grouped_features[b, :, i, k] = features[b, :, index]
    return grouped_features

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return group_points_py(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


def ball_query_py(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    r"""
    Perform ball query to find neighbors within a radius.

    Parameters
    ----------
    new_xyz : torch.Tensor
        (B, npoint, 3) tensor of query point coordinates.
    xyz : torch.Tensor
        (B, N, 3) tensor of point cloud coordinates.
    radius : float
        Radius of the ball query.
    nsample : int
        Maximum number of neighbors to sample.

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor of neighbor indices.
    """
    B, npoint, _ = new_xyz.size()
    N = xyz.size(1)

    # Compute pairwise distances between query points and point cloud points
    # new_xyz: (B, npoint, 3), xyz: (B, N, 3)
    # dist: (B, npoint, N)
    dist = torch.cdist(new_xyz, xyz)

    # Find points within the radius
    mask = dist < radius  # (B, npoint, N)

    # Initialize output indices tensor
    idx = torch.full((B, npoint, nsample), N, dtype=torch.long, device=new_xyz.device)  # N is used as a placeholder

    # For each query point, find the indices of points within the radius
    for b in range(B):
        for i in range(npoint):
            valid_idx = torch.nonzero(mask[b, i], as_tuple=True)[0]  # Indices within the radius
            if valid_idx.numel() > 0:
                # Sample up to nsample points
                if valid_idx.numel() > nsample:
                    sampled_idx = torch.randperm(valid_idx.numel(), device=new_xyz.device)[:nsample]
                    idx[b, i] = valid_idx[sampled_idx]
                else:
                    idx[b, i, :valid_idx.numel()] = valid_idx
    return idx

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return ball_query_py(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


def cylinder_query_py(
    new_xyz: torch.Tensor,
    xyz: torch.Tensor,
    rot: torch.Tensor,
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
) -> torch.Tensor:
    r"""
    Query points within a cylinder defined by radius, height range, and rotation.

    Parameters
    ----------
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the cylinder query.
    xyz : torch.Tensor
        (B, N, 3) coordinates of the features.
    rot : torch.Tensor
        (B, npoint, 9) flattened rotation matrices from cylinder frame to world frame.
    radius : float
        Radius of the cylinder.
    hmin : float
        Lower bound of the cylinder height.
    hmax : float
        Upper bound of the cylinder height.
    nsample : int
        Maximum number of samples to return per cylinder.

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) indices of the points within the cylinders.
    """
    B, npoint, _ = new_xyz.shape
    _, N, _ = xyz.shape

    # Initialize output indices tensor
    idx = torch.zeros((B, npoint, nsample), dtype=torch.long, device=xyz.device)

    for b in range(B):
        for i in range(npoint):
            # Get the center of the cylinder
            center = new_xyz[b, i]  # (3,)
            # Get the rotation matrix for this cylinder
            rotation_matrix = rot[b, i].reshape(3, 3)  # (3, 3)

            # Compute relative positions in the world frame
            rel_pos = xyz[b] - center.unsqueeze(0)  # (N, 3)

            # Transform to the cylinder frame using the rotation matrix
            rel_pos_cyl = torch.matmul(rel_pos, rotation_matrix.T)  # (N, 3)

            # Check if points are within the cylinder height range (along x-axis)
            height_mask = (rel_pos_cyl[:, 0] >= hmin) & (rel_pos_cyl[:, 0] <= hmax)  # (N,)

            # Check if points are within the radius (in y-z plane)
            radius_mask = torch.norm(rel_pos_cyl[:, 1:], dim=1) <= radius  # (N,)

            # Combine masks
            mask = height_mask & radius_mask  # (N,)

            # Get indices of points within the cylinder
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  # (valid_points,)

            # If number of valid points exceeds nsample, randomly sample nsample points
            if len(valid_idx) > nsample:
                valid_idx = valid_idx[torch.randperm(len(valid_idx))[:nsample]]

            # Pad with zeros if fewer than nsample points are found
            num_valid = len(valid_idx)
            idx[b, i, :num_valid] = valid_idx

    return idx

class CylinderQuery(Function):
    @staticmethod
    def forward(ctx, radius, hmin, hmax, nsample, xyz, new_xyz, rot):
        # type: (Any, float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the cylinders
        hmin, hmax : float
            endpoints of cylinder height in x-rotation axis
        nsample : int
            maximum number of features in the cylinders
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the cylinder query
        rot: torch.Tensor
            (B, npoint, 9) flatten rotation matrices from
                           cylinder frame to world frame

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return cylinder_query_py(new_xyz, xyz, rot, radius, hmin, hmax, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None


cylinder_query = CylinderQuery.apply


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        # type: (CylinderQueryAndGroup, float, float, float, int, bool) -> None
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)