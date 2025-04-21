// Author: chenxi-wang

#include "cylinder_query.h"
#include "utils.h"

#include <cmath>
#include <vector>
#include <iostream>

// 输入：
// - new_xyz: (b, m, 3)
// - xyz:     (b, n, 3)
// - rot:     (b, m, 3x3) 旋转矩阵 (每个中心点一个旋转)
// 输出：
// - idx:     (b, m, nsample)
void query_cylinder_point_kernel_wrapper(int b, int n, int m,
                              float radius, float hmin, float hmax,
                              int nsample,
                              const float* new_xyz,  // shape: (b, m, 3)
                              const float* xyz,      // shape: (b, n, 3)
                              const float* rot,      // shape: (b, m, 9)
                              int* idx)              // shape: (b, m, nsample)
{
    float radius2 = radius * radius;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
        const float* batch_new_xyz = new_xyz + batch_index * m * 3;
        const float* batch_xyz = xyz + batch_index * n * 3;
        const float* batch_rot = rot + batch_index * m * 9;
        int* batch_idx = idx + batch_index * m * nsample;

        for (int j = 0; j < m; ++j) {
            float new_x = batch_new_xyz[j * 3 + 0];
            float new_y = batch_new_xyz[j * 3 + 1];
            float new_z = batch_new_xyz[j * 3 + 2];

            float r0 = batch_rot[j * 9 + 0];
            float r1 = batch_rot[j * 9 + 1];
            float r2 = batch_rot[j * 9 + 2];
            float r3 = batch_rot[j * 9 + 3];
            float r4 = batch_rot[j * 9 + 4];
            float r5 = batch_rot[j * 9 + 5];
            float r6 = batch_rot[j * 9 + 6];
            float r7 = batch_rot[j * 9 + 7];
            float r8 = batch_rot[j * 9 + 8];

            int cnt = 0;
            for (int k = 0; k < n && cnt < nsample; ++k) {
                float x = batch_xyz[k * 3 + 0] - new_x;
                float y = batch_xyz[k * 3 + 1] - new_y;
                float z = batch_xyz[k * 3 + 2] - new_z;

                // Apply rotation
                float x_rot = r0 * x + r3 * y + r6 * z;
                float y_rot = r1 * x + r4 * y + r7 * z;
                float z_rot = r2 * x + r5 * y + r8 * z;

                float d2 = y_rot * y_rot + z_rot * z_rot;

                if (d2 < radius2 && x_rot > hmin && x_rot < hmax) {
                    if (cnt == 0) {
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

at::Tensor cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, const float radius, const float hmin, const float hmax,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(rot);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(rot);


  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));


    query_cylinder_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, hmin, hmax, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), rot.data<float>(), idx.data<int>());


  return idx;
}
