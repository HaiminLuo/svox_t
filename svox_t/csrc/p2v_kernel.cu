/*
 * Copyright 2022 Artemis Authors
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdint>
#include "common.cuh"
#include "data_spec_packed.cuh"

#ifndef __CUDACC__
#include <algorithm>
#include <atomic>
using std::max;
using std::min;
#endif

#define CUDA_N_THREADS 512

namespace {
void check_indices(torch::Tensor& indices) {
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());
}

namespace device {

template <typename T> __device__ __inline__ T clamp(T v, T min_v, T max_v) {
  v = v < min_v ? min_v : v;
  v = v > max_v ? max_v : v;
  return v;
}

template <typename scalar_t>
__device__ __inline__ void reduction(scalar_t * voxel_ptr) {
    return;
}

template<typename scalar_t>
__device__ __inline__ void voxelization(
    scalar_t* point, 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> voxels,
    scalar_t kernel_radius,
    scalar_t* volume_size,
    scalar_t* volume_corner
) {
    
    return;
    

}

template<typename scalar_t> 
__device__ __inline__ void cos_kernel(

) {
    return;
}

template<typename scalar_t>
__device__ __inline__ void cos_kernel_backward(

) {
    return;
}


template<typename scalar_t>
__device__ __inline__ void gaussian_kernel(

) {
    return;
}

template<typename scalar_t>
__device__ __inline__ void gaussian_kernel_backward(

) {
    return;
}

template <typename scalar_t>
__global__ void p2v_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> point_features,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> voxels,
    scalar_t kernel_radius, 
    scalar_t conv_radius,
    scalar_t* volume_size, 
    scalar_t* volume_corner
) {
    CUDA_GET_THREAD_ID(tid, points.size(0));
    int n_voxels = voxels.size(0);
    int feature_dim = point_features.size(1);

    scalar_t xyz[3] = {points[tid][0], points[tid][1], points[tid][2]};
    // scalar_t pxyz[3] = {(xyz[0] - volume_corner[0]) / voxel_size[0], (xyz[1] - volume_corner[1]) / voxel_size[1], (xyz[2] - volume_corner[2]) / voxel_size[2]};
    scalar_t voxel_size[3] = {volume_size[0] / (n_voxels - 1), volume_size[1] / (n_voxels - 1), volume_size[2] / (n_voxels - 1)};

    int min_x = clamp<int>(floor((xyz[0] - conv_radius - volume_corner[0]) / voxel_size[0]), 0, n_voxels - 1);
    int max_x = clamp<int>(ceil((xyz[0] + conv_radius - volume_corner[0]) / voxel_size[0]), 0, n_voxels - 1);
    int min_y = clamp<int>(floor((xyz[1] - conv_radius - volume_corner[1]) / voxel_size[1]), 0, n_voxels - 1);
    int max_y = clamp<int>(ceil((xyz[1] + conv_radius - volume_corner[1]) / voxel_size[1]), 0, n_voxels - 1);
    int min_z = clamp<int>(floor((xyz[2] - conv_radius - volume_corner[2]) / voxel_size[2]), 0, n_voxels - 1);
    int max_z = clamp<int>(ceil((xyz[2] + conv_radius - volume_corner[2]) / voxel_size[2]), 0, n_voxels - 1);

    int kernel_type = 1;

    for (int x = min_x; x <= max_x; ++x) 
        for (int y = min_y; y <= max_y; ++y) 
            for (int z = min_z; z <= max_z; ++z) {
                scalar_t p_voxel[3] = {x * voxel_size[0] + volume_corner[0], y * voxel_size[1] + volume_corner[1], z * voxel_size[2] + volume_corner[2]};
                scalar_t dx = xyz[0] - p_voxel[0];
                scalar_t dy = xyz[1] - p_voxel[1];
                scalar_t dz = xyz[2] - p_voxel[2];

                scalar_t r = sqrt(dx * dx + dy * dy + dz * dz);

                if (r <= conv_radius) {
                    scalar_t weight = 0.f;
                    if (kernel_type == 0) {
                        weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
                    } else {
                        weight = expf(- r * r / (2 * kernel_radius * kernel_radius));
                    }
                    scalar_t d_sigma = weight * point_features[tid][feature_dim - 1];
                    atomicAdd(&voxels[x][y][z][0], d_sigma);
                }
            }
}

template <typename scalar_t>
__global__ void p2v_kernel_backward(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> point_features,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points_grad,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> point_features_grad,
    scalar_t kernel_radius, 
    scalar_t conv_radius,
    scalar_t* volume_size, 
    scalar_t* volume_corner
 ) {
    CUDA_GET_THREAD_ID(tid, points.size(0));
    int n_voxels = grad_output.size(0);
    int feature_dim = point_features.size(1);

    scalar_t xyz[3] = {points[tid][0], points[tid][1], points[tid][2]};
    scalar_t voxel_size[3] = {volume_size[0] / (n_voxels - 1), volume_size[1] / (n_voxels - 1), volume_size[2] / (n_voxels - 1)};

    int min_x = clamp<int>(floor((xyz[0] - conv_radius - volume_corner[0]) / voxel_size[0]), 0, n_voxels - 1);
    int max_x = clamp<int>(ceil((xyz[0] + conv_radius - volume_corner[0]) / voxel_size[0]), 0, n_voxels - 1);
    int min_y = clamp<int>(floor((xyz[1] - conv_radius - volume_corner[1]) / voxel_size[1]), 0, n_voxels - 1);
    int max_y = clamp<int>(ceil((xyz[1] + conv_radius - volume_corner[1]) / voxel_size[1]), 0, n_voxels - 1);
    int min_z = clamp<int>(floor((xyz[2] - conv_radius - volume_corner[2]) / voxel_size[2]), 0, n_voxels - 1);
    int max_z = clamp<int>(ceil((xyz[2] + conv_radius - volume_corner[2]) / voxel_size[2]), 0, n_voxels - 1);

    int kernel_type = 1;

    for (int x = min_x; x <= max_x; ++x) 
        for (int y = min_y; y <= max_y; ++y) 
            for (int z = min_z; z <= max_z; ++z) {
                scalar_t p_voxel[3] = {x * voxel_size[0] + volume_corner[0], y * voxel_size[1] + volume_corner[1], z * voxel_size[2] + volume_corner[2]};
                scalar_t dx = xyz[0] - p_voxel[0];
                scalar_t dy = xyz[1] - p_voxel[1];
                scalar_t dz = xyz[2] - p_voxel[2];

                scalar_t r = sqrt(dx * dx + dy * dy + dz * dz);

                if (r <= conv_radius) {
                    scalar_t weight = 0.f;
                    if (kernel_type == 0) {
                        weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
                    } else {
                        weight = expf(- r * r / (2 * kernel_radius * kernel_radius));
                    }

                    scalar_t out_grad_value = grad_output[x][y][z][0];
                    scalar_t point_feature_value = point_features[tid][feature_dim - 1];
                    
                    // grad of point feature
                    atomicAdd(&point_features_grad[tid][0], out_grad_value * weight);

                    // grad of weight
                    scalar_t weight_grad_value = out_grad_value * point_feature_value;

                    // grad of point position
                    scalar_t point_x_grad = 0, point_y_grad = 0, point_z_grad = 0;
                    
                    if (kernel_type == 0) {
                        point_x_grad = -weight_grad_value * sin(r * M_PI / kernel_radius) *
                                        0.5 * M_PI / kernel_radius * dx /
                                        max(r, static_cast<scalar_t>(1e-10));

                        point_y_grad = -weight_grad_value * sin(r * M_PI / kernel_radius) *
                                        0.5 * M_PI / kernel_radius * dy /
                                        max(r, static_cast<scalar_t>(1e-10));

                        point_z_grad = -weight_grad_value * sin(r * M_PI / kernel_radius) *
                                        0.5 * M_PI / kernel_radius * dz /
                                        max(r, static_cast<scalar_t>(1e-10));
                    } else {
                        point_x_grad = -weight_grad_value * dx * weight / (kernel_radius * kernel_radius);
                        point_y_grad = -weight_grad_value * dy * weight / (kernel_radius * kernel_radius);
                        point_z_grad = -weight_grad_value * dz * weight / (kernel_radius * kernel_radius);
                    }
                    
                    atomicAdd(&(points_grad[tid][0]), point_x_grad);
                    atomicAdd(&(points_grad[tid][1]), point_y_grad);
                    atomicAdd(&(points_grad[tid][2]), point_z_grad);
                }
            }
}

} // namespace device

} // namespace

torch::Tensor p2v(torch::Tensor points, torch::Tensor point_features, torch::Tensor volume_corner, torch::Tensor volume_size, int n_voxels, float kernel_radius, float conv_radius) {
    check_indices(points);

    torch::Tensor voxels = torch::zeros({n_voxels, n_voxels, n_voxels, 1}, points.options());

    const auto Q = points.size(0);
    int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    AT_DISPATCH_FLOATING_TYPES(points.type(), __FUNCTION__, [&] {
        device::p2v_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                point_features.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                voxels.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                kernel_radius,
                conv_radius,
                volume_size.data<scalar_t>(),
                volume_corner.data<scalar_t>());
    });
    CUDA_CHECK_ERRORS;

    return voxels;
}

std::vector<torch::Tensor> p2v_backward(torch::Tensor grad_output, torch::Tensor points, torch::Tensor point_features, torch::Tensor volume_corner, torch::Tensor volume_size, int n_voxels, float kernel_radius, float conv_radius) {
    check_indices(points);
    const auto Q = points.size(0);
    int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor point_features_grad = torch::zeros({Q, point_features.size(1)}, points.options());
    torch::Tensor points_grad = torch::zeros({Q, points.size(1)}, points.options());

    AT_DISPATCH_FLOATING_TYPES(points.type(), __FUNCTION__, [&] {
        device::p2v_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                grad_output.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                point_features.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                points_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                point_features_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                kernel_radius,
                conv_radius,
                volume_size.data<scalar_t>(),
                volume_corner.data<scalar_t>());
    });
    CUDA_CHECK_ERRORS;

    return {points_grad, point_features_grad};
}