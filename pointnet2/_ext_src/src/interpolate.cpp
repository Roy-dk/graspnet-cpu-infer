// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "interpolate.h"
#include "utils.h"

void three_nn_kernel_wrapper(int b, int n, int m,
                             const float *unknown, // (b, n, 3)
                             const float *known,   // (b, m, 3)
                             float *dist2,         // (b, n, 3)
                             int *idx)             // (b, n, 3)
{
  for (int bs = 0; bs < b; ++bs) {
    const float *unknown_batch = unknown + bs * n * 3;
    const float *known_batch = known + bs * m * 3;
    float *dist2_batch = dist2 + bs * n * 3;
    int *idx_batch = idx + bs * n * 3;

    for (int j = 0; j < n; ++j) {
      float ux = unknown_batch[j * 3 + 0];
      float uy = unknown_batch[j * 3 + 1];
      float uz = unknown_batch[j * 3 + 2];

      float best1 = 1e40f, best2 = 1e40f, best3 = 1e40f;
      int besti1 = 0, besti2 = 0, besti3 = 0;

      for (int k = 0; k < m; ++k) {
        float x = known_batch[k * 3 + 0];
        float y = known_batch[k * 3 + 1];
        float z = known_batch[k * 3 + 2];
        float d =
            (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

        if (d < best1) {
          best3 = best2;
          besti3 = besti2;
          best2 = best1;
          besti2 = besti1;
          best1 = d;
          besti1 = k;
        } else if (d < best2) {
          best3 = best2;
          besti3 = besti2;
          best2 = d;
          besti2 = k;
        } else if (d < best3) {
          best3 = d;
          besti3 = k;
        }
      }

      dist2_batch[j * 3 + 0] = best1;
      dist2_batch[j * 3 + 1] = best2;
      dist2_batch[j * 3 + 2] = best3;

      idx_batch[j * 3 + 0] = besti1;
      idx_batch[j * 3 + 1] = besti2;
      idx_batch[j * 3 + 2] = besti3;
    }
  }
}

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, // (b, c, m)
                                      const int *idx,      // (b, n, 3)
                                      const float *weight, // (b, n, 3)
                                      float *out)          // (b, c, n)
{
  for (int bs = 0; bs < b; ++bs) {
    const float *pts = points + bs * c * m;
    const int *idx_batch = idx + bs * n * 3;
    const float *weight_batch = weight + bs * n * 3;
    float *out_batch = out + bs * c * n;

    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < n; ++j) {
        int i1 = idx_batch[j * 3 + 0];
        int i2 = idx_batch[j * 3 + 1];
        int i3 = idx_batch[j * 3 + 2];

        float w1 = weight_batch[j * 3 + 0];
        float w2 = weight_batch[j * 3 + 1];
        float w3 = weight_batch[j * 3 + 2];

        out_batch[l * n + j] =
            pts[l * m + i1] * w1 + pts[l * m + i2] * w2 + pts[l * m + i3] * w3;
      }
    }
  }
}

#include <cstring> // for memset

void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out, // (b, c, n)
                                           const int *idx,        // (b, n, 3)
                                           const float *weight,   // (b, n, 3)
                                           float *grad_points)    // (b, c, m)
{
  std::memset(grad_points, 0, sizeof(float) * b * c * m);

  for (int bs = 0; bs < b; ++bs) {
    const float *grad = grad_out + bs * c * n;
    const int *idx_batch = idx + bs * n * 3;
    const float *weight_batch = weight + bs * n * 3;
    float *grad_pts = grad_points + bs * c * m;

    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < n; ++j) {
        int i1 = idx_batch[j * 3 + 0];
        int i2 = idx_batch[j * 3 + 1];
        int i3 = idx_batch[j * 3 + 2];

        float w1 = weight_batch[j * 3 + 0];
        float w2 = weight_batch[j * 3 + 1];
        float w3 = weight_batch[j * 3 + 2];

        float g = grad[l * n + j];
        grad_pts[l * m + i1] += g * w1;
        grad_pts[l * m + i2] += g * w2;
        grad_pts[l * m + i3] += g * w3;
      }
    }
  }
}

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                          unknowns.data<float>(), knows.data<float>(),
                          dist2.data<float>(), idx.data<int>());

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  three_interpolate_kernel_wrapper(points.size(0), points.size(1),
                                   points.size(2), idx.size(1),
                                   points.data<float>(), idx.data<int>(),
                                   weight.data<float>(), output.data<float>());

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  three_interpolate_grad_kernel_wrapper(
      grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
      grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
      output.data<float>());

  return output;
}
