// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"
#include "utils.h"

#include <cstring> // for memset
#include <vector>

// 输入:
// - points: (b, c, n)
// - idx:    (b, npoints, nsample)
// 输出:
// - out:    (b, c, npoints, nsample)
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out) {
  for (int bs = 0; bs < b; ++bs) {
    const float *batch_points = points + bs * c * n;
    const int *batch_idx = idx + bs * npoints * nsample;
    float *batch_out = out + bs * c * npoints * nsample;

    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < npoints; ++j) {
        for (int k = 0; k < nsample; ++k) {
          int ii = batch_idx[j * nsample + k];
          batch_out[(l * npoints + j) * nsample + k] = batch_points[l * n + ii];
        }
      }
    }
  }
}

// 输入:
// - grad_out:     (b, c, npoints, nsample)
// - idx:          (b, npoints, nsample)
// 输出:
// - grad_points:  (b, c, n)
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
  // 初始化 grad_points 为 0
  std::memset(grad_points, 0, sizeof(float) * b * c * n);

  for (int bs = 0; bs < b; ++bs) {
    const float *batch_grad_out = grad_out + bs * c * npoints * nsample;
    const int *batch_idx = idx + bs * npoints * nsample;
    float *batch_grad_points = grad_points + bs * c * n;

    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < npoints; ++j) {
        for (int k = 0; k < nsample; ++k) {
          int ii = batch_idx[j * nsample + k];
          batch_grad_points[l * n + ii] +=
              batch_grad_out[(l * npoints + j) * nsample + k];
        }
      }
    }
  }
}

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);


  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  group_points_grad_kernel_wrapper(
      grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
      grad_out.data<float>(), idx.data<int>(), output.data<float>());

  return output;
}
