// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

#include <cmath>
// input:
//   new_xyz: (b, m, 3)
//   xyz:     (b, n, 3)
// output:
//   idx:     (b, m, nsample)
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  float radius2 = radius * radius;

  for (int batch_index = 0; batch_index < b; ++batch_index) {
    const float *batch_new_xyz = new_xyz + batch_index * m * 3;
    const float *batch_xyz = xyz + batch_index * n * 3;
    int *batch_idx = idx + batch_index * m * nsample;

    for (int j = 0; j < m; ++j) {
      float new_x = batch_new_xyz[j * 3 + 0];
      float new_y = batch_new_xyz[j * 3 + 1];
      float new_z = batch_new_xyz[j * 3 + 2];

      int cnt = 0;
      for (int k = 0; k < n && cnt < nsample; ++k) {
        float x = batch_xyz[k * 3 + 0];
        float y = batch_xyz[k * 3 + 1];
        float z = batch_xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                   (new_z - z) * (new_z - z);
        if (d2 < radius2) {
          if (cnt == 0) {
            // Pre-fill all with first found index
            for (int l = 0; l < nsample; ++l) {
              batch_idx[j * nsample + l] = k;
            }
          }
          batch_idx[j * nsample + cnt] = k;
          ++cnt;
        }
      }
    }
  }
}

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                  radius, nsample, new_xyz.data<float>(),
                                  xyz.data<float>(), idx.data<int>());

  return idx;
}
