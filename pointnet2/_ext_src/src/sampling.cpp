// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"
#include <algorithm>

void gather_points_kernel_wrapper(int b, int c, int n, int m,
                                  const float *points, const int *idx,
                                  float *out) {
  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < m; ++j) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int m,
                                       const float *grad_out, const int *idx,
                                       float *grad_points) {
  // Initialize grad_points to 0
  std::fill(grad_points, grad_points + b * c * n, 0.0f);

  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < m; ++j) {
        int a = idx[i * m + j];
        grad_points[(i * c + l) * n + a] += grad_out[(i * c + l) * m + j];
      }
    }
  }
}

#include <limits>
void furthest_point_sampling_kernel_wrapper(
    int b, int n, int m,
    const float *dataset, // shape: (b, n, 3)
    float *temp,          // shape: (b, n)
    int *idxs) {          // shape: (b, m)
  for (int bs = 0; bs < b; ++bs) {
    const float *cur_dataset = dataset + bs * n * 3;
    float *cur_temp = temp + bs * n;
    int *cur_idxs = idxs + bs * m;

    std::fill(cur_temp, cur_temp + n, std::numeric_limits<float>::max());

    int old = 0;
    cur_idxs[0] = old;

    for (int j = 1; j < m; ++j) {
      int besti = 0;
      float best = -1.0f;

      float x1 = cur_dataset[old * 3 + 0];
      float y1 = cur_dataset[old * 3 + 1];
      float z1 = cur_dataset[old * 3 + 2];

      for (int k = 0; k < n; ++k) {
        float x2 = cur_dataset[k * 3 + 0];
        float y2 = cur_dataset[k * 3 + 1];
        float z2 = cur_dataset[k * 3 + 2];
        float mag = x2 * x2 + y2 * y2 + z2 * z2;
        if (mag <= 1e-3)
          continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
                  (z2 - z1) * (z2 - z1);

        float d2 = std::min(d, cur_temp[k]);
        cur_temp[k] = d2;

        if (d2 > best) {
          best = d2;
          besti = k;
        }
      }

      old = besti;
      cur_idxs[j] = old;
    }
  }
}

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                               idx.size(1), points.data<float>(),
                               idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                    idx.size(1), grad_out.data<float>(),
                                    idx.data<int>(), output.data<float>());

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  furthest_point_sampling_kernel_wrapper(points.size(0), points.size(1),
                                         nsamples, points.data<float>(),
                                         tmp.data<float>(), output.data<int>());

  return output;
}
