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
#include "common.cuh"
#include "data_spec_packed.cuh"

#define CUDA_N_THREADS 1024
using namespace torch::indexing;

namespace {
void check_indices(torch::Tensor& indices) {
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());
}

namespace device {

template <typename scalar_t>
__device__ __inline__ scalar_t* get_tree_leaf_ptr(
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        data,
       PackedTreeSpec<scalar_t>& __restrict__ tree,
       const scalar_t* __restrict__ xyz_ind,
       int64_t* node_id=nullptr,
       int64_t* data_id=nullptr,
       int64_t* mask_out=nullptr) {
    scalar_t xyz[3] = {xyz_ind[0], xyz_ind[1], xyz_ind[2]};
    transform_coord<scalar_t>(xyz, tree.offset, tree.scaling);
    scalar_t _cube_sz;
    int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child, xyz, &_cube_sz, node_id);
    if (data_idx_ptr != nullptr && mask_out !=nullptr) {
        mask_out[*node_id] = 1;
        // printf("node id: %d\n", *node_id);
    }
    if (*data_idx_ptr >= data.size(0)) return nullptr;
    *data_id = *data_idx_ptr;
    return &data[*data_idx_ptr][0];
}

template <typename scalar_t>
__global__ void query_single_kernel(
        PackedTreeSpec<scalar_t> tree,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> values_out,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node_ids_out,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> data_ids_out,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> mask) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    // printf("fuck!\n");
    scalar_t* data_ptr = get_tree_leaf_ptr(tree.features, tree, &indices[tid][0], &node_ids_out[tid], &data_ids_out[tid], mask.data());
    // printf("fuck1!\n");
    if (data_ptr == nullptr) return;
    for (int i = 0; i < tree.features.size(1); ++i)
        values_out[tid][i] = data_ptr[i];
}

template <typename scalar_t>
__global__ void query_single_kernel_backward(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_data_out) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t* data_ptr = get_tree_leaf_ptr(grad_data_out, tree, &indices[tid][0]);
    if (data_ptr == nullptr) return;
    for (int i = 0; i < grad_output.size(1); ++i)
        atomicAdd(&data_ptr[i], grad_output[tid][i]);
}

template <typename scalar_t>
__global__ void assign_single_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> values) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));

    scalar_t* data_ptr = get_tree_leaf_ptr(tree.features, tree, &indices[tid][0]);
    if (data_ptr == nullptr) return;
    for (int i = 0; i < values.size(1); ++i)
        data_ptr[i] = values[tid][i];

}

template <typename scalar_t>
__global__ void construct_tree_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
        
    scalar_t xyz[3] = {indices[tid][0], indices[tid][1], indices[tid][2]};
    transform_coord<scalar_t>(xyz, tree.offset, tree.scaling);
    scalar_t _cube_sz;
    int32_t* data_idx_ptr = query_single_from_root<scalar_t>(tree.data, tree.child, xyz, &_cube_sz);
    *data_idx_ptr = tid;
}

template <typename scalar_t>
__global__ void warp_vertices_kernel(
       const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> skinning_weights,
       const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> joint_index, 
       torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrix_out,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices_out
    ) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    // if (tid == 2535483) printf("%d\n:", tid);
    
    const int binding_bone_num = joint_index.size(1);
    const scalar_t* skinning_weight_val = &skinning_weights[tid][0];
    const int32_t* joint_index_val = &joint_index[tid][0];

    for (int j = 0; j != binding_bone_num; ++j) {
        if (skinning_weight_val[j] > 0) 
            #pragma unroll
            for (int m = 0; m != 3; ++m) 
                #pragma unroll
                for (int n = 0; n != 4; ++n) 
                    atomicAdd(&matrix_out[tid][m][n], skinning_weight_val[j] * matrices[joint_index_val[j]][m][n]);
    }

    matrix_out[tid][3][3] = 1.f;

    #pragma unroll
    for (int i = 0 ; i != 3; ++i) {
        indices_out[tid][i] = indices[tid][0] * matrix_out[tid][i][0] + indices[tid][1] * matrix_out[tid][i][1]  +  indices[tid][2] * matrix_out[tid][i][2] + matrix_out[tid][i][3];
    }
}

template <typename scalar_t>
__global__ void warp_vertices_kernel_backward(
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices_grad_out,  
       torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrices_grad_out,
       const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> matrices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> skinning_weights,
       const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> joint_index, 
       torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_matrices,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_indices,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_skinning_weights
    ) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    // if (tid == 2535483) printf("%d\n:", tid);
    
    const int binding_bone_num = joint_index.size(1);
    const scalar_t* skinning_weight_val = &skinning_weights[tid][0];
    const int32_t* joint_index_val = &joint_index[tid][0];
    
    scalar_t matrix_out[3][4]={0,0,0,0,0,0,0,0,0,0,0,0};

    for (int j = 0; j != binding_bone_num; ++j) {
        if (skinning_weight_val[j] > 0) 
        #pragma unroll
            for (int m = 0; m != 3; ++m) 
                #pragma unroll
                for (int n = 0; n != 4; ++n) {
                    matrix_out[m][n] += skinning_weight_val[j] * matrices[joint_index_val[j]][m][n];
                    atomicAdd(&grad_matrices[joint_index_val[j]][m][n], skinning_weight_val[j] * matrices_grad_out[tid][m][n]);
                    atomicAdd(&grad_skinning_weights[tid][j], matrices[joint_index_val[j]][m][n] * matrices_grad_out[tid][m][n]);
                }
    }

    scalar_t tmp_grad_matrix[3][4]={0,0,0,0,0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int i = 0 ; i != 3; ++i) {
        grad_indices[tid][i] = indices_grad_out[tid][0] * matrix_out[0][i] + indices_grad_out[tid][1] * matrix_out[1][i]  +  indices_grad_out[tid][2] * matrix_out[2][i];
        tmp_grad_matrix[i][0] = indices_grad_out[tid][i] * indices[tid][0];
        tmp_grad_matrix[i][1] = indices_grad_out[tid][i] * indices[tid][1];
        tmp_grad_matrix[i][2] = indices_grad_out[tid][i] * indices[tid][2];
        tmp_grad_matrix[i][3] = indices_grad_out[tid][i];
    }

    for (int j = 0; j != binding_bone_num; ++j) {
        if (skinning_weight_val[j] > 0) 
        #pragma unroll
            for (int m = 0; m != 3; ++m) 
                #pragma unroll
                for (int n = 0; n != 4; ++n) {
                    atomicAdd(&grad_matrices[joint_index_val[j]][m][n], skinning_weight_val[j] * tmp_grad_matrix[m][n]);
                    atomicAdd(&grad_skinning_weights[tid][j], matrices[joint_index_val[j]][m][n] * tmp_grad_matrix[m][n]);
                }
    }

}

template <typename scalar_t>
__global__ void calc_corner_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> indexer,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {
    CUDA_GET_THREAD_ID(tid, indexer.size(0));
    const int N = tree.data.size(1);
    const auto* leaf = &indexer[tid][0];
    scalar_t* result = &output[tid][0];

    int32_t curr[4] = {(int32_t) leaf[0], (int32_t) leaf[1],
                       (int32_t) leaf[2], (int32_t) leaf[3]};
    while (true) {
        for (int i = 0; i < 3; ++i) {
            result[i] += curr[i + 1];
            result[i] /= N;
        }
        if (curr[0] == 0) break;
        curr[0] = tree.parent_depth[curr[0]][0];
        for (int i = 3; i > 0; --i) {
            curr[i] = curr[0] % N;
            curr[0] /= N;
        }
    }
}

template <typename scalar_t>
__global__ void unpack_mask_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> mask,
       torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> output) {
    CUDA_GET_THREAD_ID(tid, mask.size(0));
    const int N = tree.data.size(1);
    const int64_t m = mask[tid];

    int tmp = tid;
    if (m >= 0) {
        int64_t* result = &output[m][0];
        for (int i = 3; i > 0; --i) {
            result[i] = tmp % N;
            tmp /= N;
        }
        result[0] = tmp;
    }
}

template <typename scalar_t>
__global__ void generate_index_kernel(
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> num_hit,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> mask) {
    CUDA_GET_THREAD_ID(tid, mask.size(0));
    int64_t* m = &mask[tid];
    
    if (*m > 0) {
        *m = (int64_t)atomicAdd(&num_hit[0], 1.);
    }
}

}  // namespace device
}  // namespace

QueryResult query_vertical(TreeSpec& tree, torch::Tensor indices) {
    tree.check();
    check_indices(indices);
    DEVICE_GUARD(indices);

    const auto Q = indices.size(0), K = tree.features.size(1);
    const auto N = tree.child.size(1);
    int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);
    torch::Tensor values = torch::empty({Q, K}, indices.options());
    auto node_ids_options = at::TensorOptions()
                       .dtype(at::kLong)
                       .layout(tree.child.layout())
                       .device(tree.child.device());
    torch::Tensor node_ids = torch::empty({Q}, node_ids_options);
    torch::Tensor data_ids = torch::empty({Q}, node_ids_options);
    
    // printf("n_internal: %d, mask size: %d\n", tree.n_internal, tree.n_internal * N * N * N);
    torch::Tensor mask = torch::full({tree.n_internal * N * N * N}, -1, node_ids_options);
    torch::Tensor num_hit = torch::zeros({1}, indices.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                node_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                data_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                mask.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
    });

    blocks = CUDA_N_BLOCKS_NEEDED(mask.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::generate_index_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                num_hit.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                mask.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
    });

    // printf("hit voxels number: %d\n", num_hit.item<int>());
    torch::Tensor leaf_node = torch::zeros({num_hit.item<int>(), 4}, node_ids_options);

    blocks = CUDA_N_BLOCKS_NEEDED(mask.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::unpack_mask_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                mask.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                leaf_node.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;

    return QueryResult(values, node_ids, data_ids, leaf_node);
}

void assign_vertical(TreeSpec& tree, torch::Tensor indices, torch::Tensor values) {
    tree.check();
    check_indices(indices);
    check_indices(values);
    DEVICE_GUARD(indices);
    const int blocks = CUDA_N_BLOCKS_NEEDED(indices.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::assign_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
}

void construct_tree(TreeSpec& tree, torch::Tensor indices) {
    tree.check();
    check_indices(indices);
    DEVICE_GUARD(indices);
    const int blocks = CUDA_N_BLOCKS_NEEDED(indices.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::construct_tree_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
}

std::vector<torch::Tensor> warp_vertices(torch::Tensor matrices, torch::Tensor indices, torch::Tensor skinning_weights, torch::Tensor joint_index) {
    check_indices(indices);
    DEVICE_GUARD(indices);

    const auto Q = indices.size(0), V = indices.size(1), M = matrices.size(1);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);
    // printf("warp: %d %d\n", Q, blocks);

    torch::Tensor vertices_out = torch::zeros({Q, V}, indices.options());
    torch::Tensor matrix_out = torch::zeros({Q, M, M}, indices.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::warp_vertices_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                skinning_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                joint_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                matrix_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                vertices_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
            );
    });
    CUDA_CHECK_ERRORS;

    return {vertices_out, matrix_out};
}

torch::Tensor query_vertical_backward(
        TreeSpec& tree,
        torch::Tensor indices,
        torch::Tensor grad_output) {
    tree.check();
    DEVICE_GUARD(indices);
    const auto Q = indices.size(0),
               K = grad_output.size(1), M = tree.features.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor grad_data = torch::zeros({M, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return grad_data;
}

std::vector<torch::Tensor> warp_vertices_backward(
    torch::Tensor matrices,
    torch::Tensor indices, 
    torch::Tensor skinning_weights,
    torch::Tensor joint_index,
    torch::Tensor indices_grad_out, 
    torch::Tensor matrices_grad_out
) {
    DEVICE_GUARD(indices);
    const auto Q = indices.size(0),
               K = skinning_weights.size(1), M = matrices.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor grad_matrices = torch::zeros({M, matrices.size(1), matrices.size(2)}, matrices_grad_out.options());
    torch::Tensor grad_indices = torch::zeros({Q, indices_grad_out.size(1)}, indices_grad_out.options());
    torch::Tensor grad_skinning_weights = torch::zeros({Q, skinning_weights.size(1)}, skinning_weights.options());
    
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::warp_vertices_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                indices_grad_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                matrices_grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                skinning_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                joint_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                grad_matrices.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_skinning_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return {grad_indices, grad_matrices, grad_skinning_weights};
}

torch::Tensor calc_corners(
        TreeSpec& tree,
        torch::Tensor indexer) {
    tree.check();
    DEVICE_GUARD(indexer);
    const auto Q = indexer.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor output = torch::zeros({Q, 3}, tree.data.options());

    AT_DISPATCH_FLOATING_TYPES(tree.data.type(), __FUNCTION__, [&] {
        device::calc_corner_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indexer.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return output;
}
