/*
 * Copyright 2021 PlenOctree Authors
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
#include <vector>
#include "common.cuh"
#include "data_spec_packed.cuh"

namespace {

// Automatically choose number of CUDA threads based on HW CUDA kernel count
int cuda_n_threads = -1;
__host__ void auto_cuda_threads() {
    if (~cuda_n_threads) return;
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    const int n_cores = get_sp_cores(dev_prop);
    // Optimize number of CUDA threads per block
    if (n_cores < 2048) {
        cuda_n_threads = 256;
    } if (n_cores < 8192) {
        cuda_n_threads = 512;
    } else {
        cuda_n_threads = 512;
    }
}

namespace device {
// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __constant__ const float C4[] = {
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
};


template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm(
                scalar_t* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _normalize(
                scalar_t* dir) {
    scalar_t norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _dot3(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}


// Calculate basis functions depending on format, for given view directions
template <typename scalar_t>
__device__ __inline__ void maybe_precalc_basis(
    const int format,
    const int basis_dim,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        extra,
    const scalar_t* __restrict__ dir,
    scalar_t* __restrict__ out) {
    switch(format) {
        case FORMAT_ASG:
            {
                // UNTESTED ASG
                for (int i = 0; i < basis_dim; ++i) {
                    const auto& ptr = extra[i];
                    scalar_t S = _dot3(dir, &ptr[8]);
                    scalar_t dot_x = _dot3(dir, &ptr[2]);
                    scalar_t dot_y = _dot3(dir, &ptr[5]);
                    out[i] = S * expf(-ptr[0] * dot_x * dot_x
                                      -ptr[1] * dot_y * dot_y) / basis_dim;
                }
            }  // ASG
            break;
        case FORMAT_SG:
            {
                for (int i = 0; i < basis_dim; ++i) {
                    const auto& ptr = extra[i];
                    out[i] = expf(ptr[0] * (_dot3(dir, &ptr[1]) - 1.f)) / basis_dim;
                }
            }  // SG
            break;
        case FORMAT_SH:
            {
                out[0] = C0;
                const scalar_t x = dir[0], y = dir[1], z = dir[2];
                const scalar_t xx = x * x, yy = y * y, zz = z * z;
                const scalar_t xy = x * y, yz = y * z, xz = x * z;
                switch (basis_dim) {
                    case 25:
                        out[16] = C4[0] * xy * (xx - yy);
                        out[17] = C4[1] * yz * (3 * xx - yy);
                        out[18] = C4[2] * xy * (7 * zz - 1.f);
                        out[19] = C4[3] * yz * (7 * zz - 3.f);
                        out[20] = C4[4] * (zz * (35 * zz - 30) + 3);
                        out[21] = C4[5] * xz * (7 * zz - 3);
                        out[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
                        out[23] = C4[7] * xz * (xx - 3 * yy);
                        out[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
                        [[fallthrough]];
                    case 16:
                        out[9] = C3[0] * y * (3 * xx - yy);
                        out[10] = C3[1] * xy * z;
                        out[11] = C3[2] * y * (4 * zz - xx - yy);
                        out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                        out[13] = C3[4] * x * (4 * zz - xx - yy);
                        out[14] = C3[5] * z * (xx - yy);
                        out[15] = C3[6] * x * (xx - 3 * yy);
                        [[fallthrough]];
                    case 9:
                        out[4] = C2[0] * xy;
                        out[5] = C2[1] * yz;
                        out[6] = C2[2] * (2.0 * zz - xx - yy);
                        out[7] = C2[3] * xz;
                        out[8] = C2[4] * (xx - yy);
                        [[fallthrough]];
                    case 4:
                        out[1] = -C1 * y;
                        out[2] = C1 * z;
                        out[3] = -C1 * x;
                }
            }  // SH
            break;

        default:
            // Do nothing
            break;
    }  // switch
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template <typename scalar_t>
__device__ __inline__ void _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax) {
    // Perform DDA for 1 iteration on a unit cube
    scalar_t t1, t2;
    *tmin = 0.0f;
    *tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * invdir[i];
        t2 = t1 +  invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}


template <typename scalar_t>
__device__ __inline__ void trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);
    const int out_data_dim = out.size(0) - 1;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = opt.background_brightness;
        }
        out[out_data_dim] = 0;
        return;
    } else {
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        scalar_t pos[3];
        scalar_t basis_fn[25];
        maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim,
                tree.extra_data, ray.vdir, basis_fn);

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j]; 
            }

            int64_t node_id;
            int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);
            
            scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);
            
            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;
            // printf("data_idx: %d, cub_size: %f, feature_size: %d, sigma: %f, delta: %f, data_dim: %d\n", *data_idx_ptr, cube_sz, tree.features.size(0), sigma, delta_t, data_dim);
            if (sigma > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);
                
                if (tree.transformation_matrices.size(0) != 0) {
                    scalar_t ray_dir[3] = {ray.vdir[0], ray.vdir[1], ray.vdir[2]};
                    ray_dir[0] = tree.transformation_matrices[*data_idx_ptr][0][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][0][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][0][2] * ray.vdir[2];
                    ray_dir[1] = tree.transformation_matrices[*data_idx_ptr][1][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][1][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][1][2] * ray.vdir[2];
                    ray_dir[2] = tree.transformation_matrices[*data_idx_ptr][2][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][2][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][2][2] * ray.vdir[2];
                    
                    maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim,
                        tree.extra_data, ray_dir, basis_fn);
                }

                if (opt.format != FORMAT_RGBA) {
                    for (int t = 0; t < out_data_dim; ++ t) {
                        int off = t * opt.basis_dim;
                        scalar_t tmp = 0.0;
                        for (int i = opt.min_comp; i <= opt.max_comp; ++i) {
                            tmp += basis_fn[i] * tree_val[off + i];
                        }
                        out[t] += weight / (1.0 + expf(-tmp));
                    }
                } else {
                    for (int j = 0; j < out_data_dim; ++j) {
                        out[j] += weight / (1.0 + expf(-tree_val[j]));
                    }
                }
                light_intensity *= att;

                if (tree.weight_accum != nullptr) {
                    tree.weight_accum[node_id] += weight;
                }

                if (light_intensity <= opt.stop_thresh) {
                    // Full opacity, stop
                    scalar_t scale = 1.0 / (1.0 - light_intensity);
                    for (int j = 0; j != out_data_dim; ++j) out[j] *= scale;
                    out[out_data_dim] = 1 - light_intensity;
                    return;
                }
            }
            t += delta_t;
        }
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] += light_intensity * opt.background_brightness;
        }
        out[out_data_dim] = 1 - light_intensity;
    }
}

template <typename scalar_t>
__device__ __inline__ void trace_ray_backward(
    PackedTreeSpec<scalar_t>& __restrict__ tree,
    const torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t>
        grad_output,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);
    const int out_data_dim = grad_output.size(0) - 1;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        scalar_t pos[3];
        scalar_t basis_fn[25];
        maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim, tree.extra_data,
                ray.vdir, basis_fn);

        scalar_t accum = 0.0;
        scalar_t light_intensity_ray = 0.0;
        // PASS 1
        {
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = ray.origin[j] + t * ray.dir[j];

                int32_t* data_idx_ptr = query_single_from_root<scalar_t>(
                        tree.data, tree.child, pos, &cube_sz);
                scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + opt.step_size;
                const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;

                if (sigma > 0.0) {
                    // Reuse offset on gradient
                    const int64_t curr_leaf_offset = tree_val - tree.features.data();
                    scalar_t* grad_tree_val = grad_data_out.data() + curr_leaf_offset;

                    if (tree.transformation_matrices.size(0) != 0) {
                        scalar_t ray_dir[3] = {ray.vdir[0], ray.vdir[1], ray.vdir[2]};
                        ray_dir[0] = tree.transformation_matrices[*data_idx_ptr][0][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][0][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][0][2] * ray.vdir[2];
                        ray_dir[1] = tree.transformation_matrices[*data_idx_ptr][1][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][1][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][1][2] * ray.vdir[2];
                        ray_dir[2] = tree.transformation_matrices[*data_idx_ptr][2][0] * ray.vdir[0] + tree.transformation_matrices[*data_idx_ptr][2][1] * ray.vdir[1] + tree.transformation_matrices[*data_idx_ptr][2][2] * ray.vdir[2];
                        
                        maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim,
                            tree.extra_data, ray_dir, basis_fn);
                    }

                    att = expf(-delta_t * sigma * delta_scale);
                    const scalar_t weight = light_intensity * (1.f - att);

                    scalar_t total_color = 0.f;
                    if (opt.format != FORMAT_RGBA) {
                        for (int t = 0; t < out_data_dim; ++ t) {
                            int off = t * opt.basis_dim;
                            scalar_t tmp = 0.0;
                            for (int i = opt.min_comp; i <= opt.max_comp; ++i) {
                                tmp += basis_fn[i] * tree_val[off + i];
                            }
                            const scalar_t sigmoid = 1.0 / (1.0 + expf(-tmp));
                            const scalar_t grad_sigmoid = sigmoid * (1.0 - sigmoid);
                            for (int i = opt.min_comp; i <= opt.max_comp; ++i) {
                                const scalar_t toadd = weight * basis_fn[i] *
                                    grad_sigmoid * grad_output[t];
                                atomicAdd(&grad_tree_val[off + i],
                                        toadd);
                            }
                            total_color += sigmoid * grad_output[t];
                        }
                    } else {
                        for (int j = 0; j < out_data_dim; ++j) {
                            const scalar_t sigmoid = 1.0 / (1.0 + expf(-tree_val[j]));
                            const scalar_t toadd = weight * sigmoid * (
                                    1.f - sigmoid) * grad_output[j];
                            atomicAdd(&grad_tree_val[j], toadd);
                            total_color += sigmoid * grad_output[j];
                        }
                    }
                    light_intensity *= att;
                    accum += weight * total_color;
                }
                t += delta_t;
            }
            scalar_t total_grad = 0.f;
            for (int j = 0; j < out_data_dim; ++j)
                total_grad += grad_output[j];
            accum += light_intensity * opt.background_brightness * total_grad;
            light_intensity_ray = light_intensity;
        }
        // PASS 2
        {
            // scalar_t accum_lo = 0.0;
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = ray.origin[j] + t * ray.dir[j];
                int32_t* data_idx_ptr = query_single_from_root<scalar_t>(
                        tree.data, tree.child, pos, &cube_sz);
                scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];
                
                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + opt.step_size;
                const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;

                if (sigma > 0.0) {
                    // Reuse offset on gradient
                    const int64_t curr_leaf_offset = tree_val - tree.features.data();
                    scalar_t* grad_tree_val = grad_data_out.data() + curr_leaf_offset;

                    att = expf(-delta_t * sigma * delta_scale);
                    const scalar_t weight = light_intensity * (1.f - att);

                    scalar_t total_color = 0.f;
                    if (opt.format != FORMAT_RGBA) {
                        for (int t = 0; t < out_data_dim; ++ t) {
                            int off = t * opt.basis_dim;
                            scalar_t tmp = 0.0;
                            for (int i = opt.min_comp; i <= opt.max_comp; ++i) {
                                tmp += basis_fn[i] * tree_val[off + i];
                            }
                            total_color += 1.0 / (1.0 + expf(-tmp)) * grad_output[t];
                        }
                    } else {
                        for (int j = 0; j < out_data_dim; ++j) {
                            total_color += 1.0 / (1.0 + expf(-tree_val[j])) * grad_output[j];
                        }
                    }
                    light_intensity *= att;
                    accum -= weight * total_color;
                    // atomicAdd(
                    //        &grad_tree_val[out_data_dim],
                    //        delta_t * delta_scale * (
                    //            total_color * light_intensity - accum)
                    //        );
                    atomicAdd(
                            &grad_tree_val[data_dim - 1],
                            delta_t * delta_scale * (
                                total_color * light_intensity - accum) + delta_t * delta_scale * grad_output[out_data_dim] * light_intensity_ray
                            );
                }
                t += delta_t;
            }
        }
    }
}


template <typename scalar_t>
__device__ __inline__ void opacity_trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        out[0] = 0.f;
        return;
    } else {
        out[0] = 0.f;
        scalar_t pos[3];
        
        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
            }

            int64_t node_id;
            int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);
            
            scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);
            
            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;
            if (sigma > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                light_intensity *= att;
                if (light_intensity <= opt.stop_thresh) {
                    // Full opacity, stop
                    out[0] = 1 - light_intensity;
                    return;
                }
            }
            t += delta_t;
        }
        out[0] = 1 - light_intensity;
    }
}

template <typename scalar_t>
__device__ __inline__ void opacity_trace_ray_backward(
    PackedTreeSpec<scalar_t>& __restrict__ tree,
    const torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t>
        grad_output,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        scalar_t pos[3];

        scalar_t accum = 0.0;
        scalar_t light_intensity_ray = 0.0;
        // PASS 1
        {
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = ray.origin[j] + t * ray.dir[j];

                int32_t* data_idx_ptr = query_single_from_root<scalar_t>(
                        tree.data, tree.child, pos, &cube_sz);
                scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + opt.step_size;
                const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;

                if (sigma > 0.0) {
                    // Reuse offset on gradient
                    att = expf(-delta_t * sigma * delta_scale);
                    light_intensity *= att;
                }
                t += delta_t;
            }
            light_intensity_ray = light_intensity;
        }
        // PASS 2
        {
            // scalar_t accum_lo = 0.0;
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = ray.origin[j] + t * ray.dir[j];
                int32_t* data_idx_ptr = query_single_from_root<scalar_t>(
                        tree.data, tree.child, pos, &cube_sz);
                scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];
                
                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + opt.step_size;
                const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;

                if (sigma > 0.0) {
                    // Reuse offset on gradient
                    const int64_t curr_leaf_offset = tree_val - tree.features.data();
                    scalar_t* grad_tree_val = grad_data_out.data() + curr_leaf_offset;
                    att = expf(-delta_t * sigma * delta_scale);
                    atomicAdd(
                            &grad_tree_val[data_dim - 1],
                            delta_t * delta_scale * grad_output[0] * light_intensity_ray
                            );
                }
                t += delta_t;
            }
        }
    }
}


template <typename scalar_t>
__global__ void render_ray_kernel(
        PackedTreeSpec<scalar_t> tree,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    // printf("matrices: %d\n", tree.transformation_matrices.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        out[tid]);
}


template <typename scalar_t>
__global__ void render_ray_backward_kernel(
    PackedTreeSpec<scalar_t> tree,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    trace_ray_backward<scalar_t>(
        tree,
        grad_output[tid],
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        grad_data_out);
}


// motion feature ray tracing 
template <typename scalar_t>
__device__ __inline__ void motion_trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out, 
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> depth_out,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> hit_point_out,
        torch::TensorAccessor<int64_t, 1, torch::RestrictPtrTraits, int32_t> data_idx_out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int data_dim = tree.features.size(1);
    const int out_data_dim = out.size(0);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        depth_out[0] = 0.f;
        hit_point_out[0] = 0.f;
        hit_point_out[1] = 0.f;
        hit_point_out[2] = 0.f;
        return;
    } else {
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        scalar_t pos[3];

        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
            }

            int64_t node_id;
            int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);
            
            scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);
            
            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;
            // printf("data_idx: %d, cub_size: %f, feature_size: %d, sigma: %f, delta: %f, data_dim: %d\n", *data_idx_ptr, cube_sz, tree.features.size(0), sigma, delta_t, data_dim);
            if (sigma > opt.sigma_thresh) {
                transform_coord_world<scalar_t>(pos, tree.offset, tree.scaling);

                hit_point_out[0] = pos[0];
                hit_point_out[1] = pos[1];
                hit_point_out[2] = pos[2];

                depth_out[0] = t * delta_scale;
                scalar_t dis[3];
                for (int i = 0 ; i != out_data_dim ; ++i) {
                    dis[0] = pos[0] - tree.extra_data[i][0];
                    dis[1] = pos[1] - tree.extra_data[i][1];
                    dis[2] = pos[2] - tree.extra_data[i][2];
                    out[i] = _norm(dis);
                }
                data_idx_out[0] = *data_idx_ptr;

                return;
            }
            t += delta_t;
        }
    }
}

// depth ray tracing 
template <typename scalar_t>
__device__ __inline__ void depth_trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> depth_out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int data_dim = tree.features.size(1);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        depth_out[0] = 0.f; 
        return;
    } else {
        depth_out[0] = 0.f; 
        scalar_t pos[3];

        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
            }

            int64_t node_id;
            int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);
            
            scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];

            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);
            
            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;
            if (sigma > opt.sigma_thresh) {
                transform_coord_world<scalar_t>(pos, tree.offset, tree.scaling);
                depth_out[0] = delta_scale * t;
                return;
            }
            t += delta_t;
        }
    }
}

template <typename scalar_t>
__global__ void motion_render_ray_kernel(
        PackedTreeSpec<scalar_t> tree,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
            out,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
            depth_out,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
            hit_point_out,
        torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits>
            data_idx_out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    motion_trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        out[tid], 
        depth_out[tid],
        hit_point_out[tid],
        data_idx_out[tid]
    );
}


template <typename scalar_t>
__global__ void depth_render_ray_kernel(
        PackedTreeSpec<scalar_t> tree,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
            depth_out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    depth_trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        depth_out[tid]
    );
}


template <typename scalar_t>
__device__ __inline__ void motion_feature_trace_ray(
        PackedTreeSpec<scalar_t>& __restrict__ tree,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);
    const int joint_num = tree.joint_features.size(0);
    const int joint_feature_dim = tree.joint_features.size(1);
    const int binding_bone_num = tree.joint_index.size(1);
    const int out_data_dim = out.size(0);
    
    // printf("const ints: %d, %d, %d, %d, %d\n",  data_dim, joint_num, joint_feature_dim, out_data_dim, binding_bone_num);

    const int tmp_data_dim = 32;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
            // out[j] = opt.background_brightness;
        }
        return;
    } else {
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        scalar_t pos[3];
        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = ray.origin[j] + t * ray.dir[j];
            }

            int64_t node_id;
            int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child,
                        pos, &cube_sz, tree.weight_accum != nullptr ? &node_id : nullptr);
            
            scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];
            scalar_t* skinning_weight_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.skinning_weights[*data_idx_ptr][0];
            int32_t* joint_index_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.joint_index[*data_idx_ptr][0];

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);
            
            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;
            // printf("data_idx: %d, cub_size: %f, feature_size: %d, sigma: %f, delta: %f, data_dim: %d\n", *data_idx_ptr, cube_sz, tree.features.size(0), sigma, delta_t, data_dim);
            // printf("motion features: %d, dim: %d\n", binding_bone_num, joint_feature_dim);
            if (sigma > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                scalar_t pos_joint_feature[tmp_data_dim];
                for (int j = 0; j != out_data_dim ; ++j) pos_joint_feature[j] = 0.f;
                for (int j = 0; j != binding_bone_num; ++j) {
                    if (skinning_weight_val[j] > 0) 
                        #pragma unroll
                        for (int k = 0; k != out_data_dim; ++k) pos_joint_feature[k] += skinning_weight_val[j] * tree.joint_features[joint_index_val[j]][k];                      
                }
                
                for (int j = 0; j != out_data_dim; ++j) {
                    out[j] += weight / ( 1.0 + expf(-pos_joint_feature[j]));
                }

                light_intensity *= att;

                if (light_intensity <= opt.stop_thresh) {
                    // Full opacity, stop
                    scalar_t scale = 1.0 / (1.0 - light_intensity);
                    for (int j = 0; j != out_data_dim; ++j) out[j] *= scale;
                    return;
                }
            }
            t += delta_t;
        }
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] += light_intensity * opt.background_brightness;
        }
    }
}

template <typename scalar_t>
__device__ __inline__ void motion_feature_trace_ray_backward(
    PackedTreeSpec<scalar_t>& __restrict__ tree,
    const torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t>
        grad_output,
        SingleRaySpec<scalar_t> ray,
        RenderOptions& __restrict__ opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out) {
    const scalar_t delta_scale = _get_delta_scale(tree.scaling, ray.dir);

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = tree.child.size(1);
    const int data_dim = tree.features.size(1);
    const int joint_num = tree.joint_features.size(0);
    const int joint_feature_dim = tree.joint_features.size(1);
    const int binding_bone_num = tree.joint_index.size(1);
    const int out_data_dim = grad_output.size(0);

    const int tmp_data_dim = 32;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (ray.dir[i] + 1e-9);
    }
    _dda_unit(ray.origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        scalar_t pos[3];
        // PASS 1
        {
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = ray.origin[j] + t * ray.dir[j];

                int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child, pos, &cube_sz);
                scalar_t* tree_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.features[*data_idx_ptr][0];
                scalar_t* skinning_weight_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.skinning_weights[*data_idx_ptr][0];
                int32_t* joint_index_val = *data_idx_ptr >= tree.features.size(0) ? nullptr : &tree.joint_index[*data_idx_ptr][0];

                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + opt.step_size;
                const scalar_t sigma = tree_val != nullptr ? tree_val[data_dim - 1] : 0.0;

                if (sigma > 0.0) {
                    scalar_t grad_tmp[tmp_data_dim];

                    att = expf(-delta_t * sigma * delta_scale);
                    const scalar_t weight = light_intensity * (1.f - att);

                    scalar_t pos_joint_feature[tmp_data_dim];
                    for (int j = 0; j != tmp_data_dim ; ++j) pos_joint_feature[j] = 0.f;
                    for (int j = 0; j != binding_bone_num ; ++j) {
                        if (skinning_weight_val[j] > 0) 
                            for (int k = 0; k != out_data_dim; ++k) pos_joint_feature[k] += skinning_weight_val[j] * tree.joint_features[joint_index_val[j]][k];                      
                    }

                    for (int j = 0; j != out_data_dim; ++j) {
                        const scalar_t sigmoid = 1.0 / (1.0 + expf(-pos_joint_feature[j]));
                        const scalar_t toadd = weight * sigmoid * (
                                    1.f - sigmoid) * grad_output[j];
                        grad_tmp[j] += toadd;
                    }

                    for (int j = 0; j != binding_bone_num ; ++j) {
                        if (skinning_weight_val[j] > 0) 
                            for (int k = 0; k != out_data_dim; ++k) atomicAdd(&grad_data_out[joint_index_val[j]][k], skinning_weight_val[j] * grad_tmp[j]);                      
                    }

                    light_intensity *= att;
                }
                t += delta_t;
            }
        }
    }
}

template <typename scalar_t>
__global__ void moition_feature_render_ray_kernel(
        PackedTreeSpec<scalar_t> tree,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    // printf("tid: %d\n", tid);
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    motion_feature_trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        out[tid]);
}


template <typename scalar_t>
__global__ void moition_feature_render_ray_backward_kernel(
    PackedTreeSpec<scalar_t> tree,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    motion_feature_trace_ray_backward<scalar_t>(
        tree,
        grad_output[tid],
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        grad_data_out);
}

template <typename scalar_t>
__global__ void opacity_render_ray_kernel(
        PackedTreeSpec<scalar_t> tree,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    // printf("tid: %d\n", tid);
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    opacity_trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        out[tid]);
}

template <typename scalar_t>
__global__ void opacity_render_ray_backward_kernel(
    PackedTreeSpec<scalar_t> tree,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
        PackedRaysSpec<scalar_t> rays,
        RenderOptions opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out
        ) {
    CUDA_GET_THREAD_ID(tid, rays.origins.size(0));
    scalar_t origin[3] = {rays.origins[tid][0], rays.origins[tid][1], rays.origins[tid][2]};
    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    scalar_t dir[3] = {rays.dirs[tid][0], rays.dirs[tid][1], rays.dirs[tid][2]};
    opacity_trace_ray_backward<scalar_t>(
        tree,
        grad_output[tid],
        SingleRaySpec<scalar_t>{origin, dir, &rays.vdirs[tid][0]},
        opt,
        grad_data_out);
}



template <typename scalar_t>
__device__ __inline__ void cam2world_ray(
    int ix, int iy,
    scalar_t* dir,
    scalar_t* origin,
    const PackedCameraSpec<scalar_t>& __restrict__ cam) {
    scalar_t x = (ix - 0.5 * cam.width) / cam.fx;
    scalar_t y = -(iy - 0.5 * cam.height) / cam.fy;
    scalar_t z = sqrtf(x * x + y * y + 1.0);
    x /= z; y /= z; z = -1.0f / z;
    dir[0] = cam.c2w[0][0] * x + cam.c2w[0][1] * y + cam.c2w[0][2] * z;
    dir[1] = cam.c2w[1][0] * x + cam.c2w[1][1] * y + cam.c2w[1][2] * z;
    dir[2] = cam.c2w[2][0] * x + cam.c2w[2][1] * y + cam.c2w[2][2] * z;
    origin[0] = cam.c2w[0][3]; origin[1] = cam.c2w[1][3]; origin[2] = cam.c2w[2][3];
}


template <typename scalar_t>
__host__ __device__ __inline__ static void maybe_world2ndc(
        RenderOptions& __restrict__ opt,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen, scalar_t near = 1.f) {
    if (opt.ndc_width < 0)
        return;
    scalar_t t = -(near + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * opt.ndc_focal) / opt.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * opt.ndc_focal) / opt.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 * near / cen[2];

    cen[0] = -((2 * opt.ndc_focal) / opt.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * opt.ndc_focal) / opt.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 * near / cen[2];

    _normalize(dir);
}


template <typename scalar_t>
__global__ void render_image_kernel(
    PackedTreeSpec<scalar_t> tree,
    PackedCameraSpec<scalar_t> cam,
    RenderOptions opt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    maybe_world2ndc(opt, dir, origin);

    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    trace_ray<scalar_t>(
        tree,
        SingleRaySpec<scalar_t>{origin, dir, vdir},
        opt,
        out[iy][ix]);
}

template <typename scalar_t>
__global__ void render_image_backward_kernel(
    PackedTreeSpec<scalar_t> tree,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    PackedCameraSpec<scalar_t> cam,
    RenderOptions opt,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_data_out) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    maybe_world2ndc(opt, dir, origin);

    transform_coord<scalar_t>(origin, tree.offset, tree.scaling);
    trace_ray_backward<scalar_t>(
        tree,
        grad_output[iy][ix],
        SingleRaySpec<scalar_t>{origin, dir, vdir},
        opt,
        grad_data_out);
}

template <typename scalar_t>
__device__ __inline__ void grid_trace_ray(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        data,
        const scalar_t* __restrict__ origin,
        const scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        scalar_t step_size,
        scalar_t delta_scale,
        scalar_t sigma_thresh,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grid_weight,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grid_hit) {
    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int reso = data.size(0);
    scalar_t* grid_weight_val = grid_weight.data();
    scalar_t* grid_hit_val = grid_hit.data();

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (dir[i] + 1e-9);
    }
    _dda_unit(origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        scalar_t pos[3];

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz = reso;
        int32_t u, v, w, node_id;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = origin[j] + t * dir[j];
            }

            clamp_coord<scalar_t>(pos);
            pos[0] *= reso;
            pos[1] *= reso;
            pos[2] *= reso;
            u = floor(pos[0]);
            v = floor(pos[1]);
            w = floor(pos[2]);
            pos[0] -= u;
            pos[1] -= v;
            pos[2] -= w;
            node_id = u * reso * reso + v * reso + w;

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + step_size;
            const scalar_t sigma = data[u][v][w];
            if (sigma > sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);
                light_intensity *= att;

                atomicMax(&grid_weight_val[node_id], weight);
                atomicAdd(&grid_hit_val[node_id], (scalar_t) 1.0);
            }
            t += delta_t;
        }
    }
}

template <typename scalar_t>
__global__ void grid_weight_render_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        data,
    PackedCameraSpec<scalar_t> cam,
    RenderOptions opt,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ scaling,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grid_weight,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grid_hit) {
    CUDA_GET_THREAD_ID(tid, cam.width * cam.height);
    int iy = tid / cam.width, ix = tid % cam.width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, dir, origin, cam);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    maybe_world2ndc(opt, dir, origin);

    transform_coord<scalar_t>(origin, offset, scaling);
    const scalar_t delta_scale = _get_delta_scale(scaling, dir);
    grid_trace_ray<scalar_t>(
        data,
        origin,
        dir,
        vdir,
        opt.step_size,
        delta_scale,
        opt.sigma_thresh,
        grid_weight,
        grid_hit);
}



}  // namespace device


// Compute RGB output dimension from input dimension & SH degree
__host__ int get_out_data_dim(int format, int basis_dim, int in_data_dim) {
    if (format != FORMAT_RGBA) {
        return (in_data_dim - 1) / basis_dim + 1;
    } else {
        return in_data_dim;
    }
}

}  // namespace

torch::Tensor volume_render(TreeSpec& tree, RaysSpec& rays, RenderOptions& opt) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.features);
    const auto Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(opt.format, opt.basis_dim, tree.features.size(1));
    torch::Tensor result = torch::zeros({Q, out_data_dim}, rays.origins.options());
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, rays, opt,
                    result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_image(TreeSpec& tree, CameraSpec& cam, RenderOptions& opt) {
    tree.check();
    cam.check();
    DEVICE_GUARD(tree.features);
    const size_t Q = size_t(cam.width) * cam.height;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(opt.format, opt.basis_dim, tree.features.size(1));
    torch::Tensor result = torch::zeros({cam.height, cam.width, out_data_dim},
            tree.data.options());

    AT_DISPATCH_FLOATING_TYPES(tree.data.type(), __FUNCTION__, [&] {
            device::render_image_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, cam, opt,
                    result.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_backward(
    TreeSpec& tree, RaysSpec& rays,
    RenderOptions& opt,
    torch::Tensor grad_output) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.features);

    const int Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(opt.format, opt.basis_dim, tree.features.size(1));
    torch::Tensor result = torch::zeros_like(tree.features);
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::render_ray_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                tree,
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                rays,
                opt,
                result.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor volume_render_image_backward(TreeSpec& tree, CameraSpec& cam,
                                           RenderOptions& opt,
                                           torch::Tensor grad_output) {
    tree.check();
    cam.check();
    DEVICE_GUARD(tree.features);

    const size_t Q = size_t(cam.width) * cam.height;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(opt.format, opt.basis_dim, tree.features.size(1));
    torch::Tensor result = torch::zeros_like(tree.features);

    AT_DISPATCH_FLOATING_TYPES(tree.data.type(), __FUNCTION__, [&] {
            device::render_image_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                tree,
                grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                cam,
                opt,
                result.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

std::vector<torch::Tensor> grid_weight_render(
    torch::Tensor data, CameraSpec& cam, RenderOptions& opt,
    torch::Tensor offset, torch::Tensor scaling) {
    cam.check();
    DEVICE_GUARD(data);
    const size_t Q = size_t(cam.width) * cam.height;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor grid_weight = torch::zeros_like(data);
    torch::Tensor grid_hit = torch::zeros_like(data);

    AT_DISPATCH_FLOATING_TYPES(data.type(), __FUNCTION__, [&] {
            device::grid_weight_render_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                data.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                cam,
                opt,
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                grid_weight.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grid_hit.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return {grid_weight, grid_hit};
}

std::vector<torch::Tensor> motion_render(TreeSpec& tree, RaysSpec& rays, RenderOptions& opt) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.extra_data);
    const auto Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int motion_feature_dim = tree.extra_data.size(0);
    torch::Tensor result = torch::zeros({Q, motion_feature_dim}, rays.origins.options());
    torch::Tensor depth = torch::zeros({Q, 1}, rays.origins.options());
    torch::Tensor hit_point = torch::zeros({Q, 3}, rays.origins.options());
    torch::Tensor data_idx = torch::zeros({Q, 1}, rays.origins.options()).to(torch::kInt64);
    
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::motion_render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, rays, opt,
                    result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    depth.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    hit_point.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    data_idx.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return {result, depth, hit_point, data_idx};
}

torch::Tensor render_depth(TreeSpec& tree, RaysSpec& rays, RenderOptions& opt) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.extra_data);
    const auto Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor depth = torch::zeros({Q, 1}, rays.origins.options());
    
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::depth_render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, rays, opt,
                    depth.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return depth;
}

torch::Tensor motion_feature_render(TreeSpec& tree, RaysSpec& rays, RenderOptions& opt) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.joint_features);
    const auto Q = rays.origins.size(0);

    auto_cuda_threads();
    // cuda_n_threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = tree.joint_features.size(1);
    torch::Tensor result = torch::zeros({Q, out_data_dim}, rays.origins.options());
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::moition_feature_render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, rays, opt,
                    result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}


torch::Tensor motion_feature_render_backward(
    TreeSpec& tree, RaysSpec& rays,
    RenderOptions& opt,
    torch::Tensor grad_output) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.joint_features);

    const int Q = rays.origins.size(0);

    auto_cuda_threads();

    // cuda_n_threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = tree.joint_features.size(1);
    torch::Tensor result = torch::zeros_like(tree.joint_features);
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::moition_feature_render_ray_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                tree,
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                rays,
                opt,
                result.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor opacity_render(TreeSpec& tree, RaysSpec& rays, RenderOptions& opt) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.features);
    const auto Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = 1;
    torch::Tensor result = torch::zeros({Q, out_data_dim}, rays.origins.options());
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::opacity_render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                    tree, rays, opt,
                    result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor opacity_render_backward(
    TreeSpec& tree, RaysSpec& rays,
    RenderOptions& opt,
    torch::Tensor grad_output) {
    tree.check();
    rays.check();
    DEVICE_GUARD(tree.features);

    const int Q = rays.origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    torch::Tensor result = torch::zeros_like(tree.features);
    AT_DISPATCH_FLOATING_TYPES(rays.origins.type(), __FUNCTION__, [&] {
            device::render_ray_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                tree,
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                rays,
                opt,
                result.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}
