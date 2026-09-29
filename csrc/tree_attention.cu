// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Carlo Cetrone. All rights reserved.
// CUDA kernel for generating 2D Tree Attention causal masks.

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace cautious_decoding {

// CUDA Kernel: Builds the 2D boolean tree attention mask [N, N]
// mask[i, j] = 1 if j is an ancestor of i (or i == j), else 0
__global__ void build_tree_attention_mask_kernel(
    const int* __restrict__ parent_indices, // [num_tokens], parent of node i, or -1 for root
    bool* __restrict__ mask,                // [num_tokens, num_tokens]
    int num_tokens
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // query index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // key index

    if (i >= num_tokens || j >= num_tokens) return;

    if (i == j) {
        mask[i * num_tokens + j] = true;
        return;
    }

    // Traverse upwards from node i to check if node j is an ancestor
    bool is_ancestor = false;
    int curr = parent_indices[i];
    while (curr >= 0) {
        if (curr == j) {
            is_ancestor = true;
            break;
        }
        curr = parent_indices[curr];
    }

    mask[i * num_tokens + j] = is_ancestor;
}

torch::Tensor build_tree_attention_mask_cuda(
    torch::Tensor parent_indices // [num_tokens] on CUDA
) {
    int num_tokens = parent_indices.size(0);
    auto mask = torch::zeros({num_tokens, num_tokens}, torch::dtype(torch::kBool).device(parent_indices.device()));

    dim3 threads(16, 16);
    dim3 blocks((num_tokens + 15) / 16, (num_tokens + 15) / 16);

    build_tree_attention_mask_kernel<<<blocks, threads>>>(
        parent_indices.data_ptr<int>(),
        mask.data_ptr<bool>(),
        num_tokens
    );

    return mask;
}

} // namespace cautious_decoding
