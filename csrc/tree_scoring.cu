// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Carlo Cetrone. All rights reserved.
// High-performance CUDA kernels for on-device tree perplexity scoring and pruning.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <float.h>

namespace cautious_decoding {

// CUDA Kernel: Computes path log-probabilities and performs parallel reduction to find best path
__global__ void compute_path_perplexity_kernel(
    const float* __restrict__ candidate_logprobs, // [num_parent_nodes, breadth]
    const int* __restrict__ path_node_indices,    // [num_paths, depth]
    const int* __restrict__ path_child_indices,   // [num_paths, depth]
    float* __restrict__ path_ppls,                // [num_paths]
    int num_paths,
    int depth,
    int breadth
) {
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= num_paths) return;

    float sum_logprob = 0.0f;
    for (int d = 0; d < depth; ++d) {
        int node_idx = path_node_indices[path_idx * depth + d];
        int child_idx = path_child_indices[path_idx * depth + d];
        float lp = candidate_logprobs[node_idx * breadth + child_idx];
        sum_logprob += lp;
    }

    // PPL = exp(-1/depth * sum_logprob)
    float avg_neg_lp = -sum_logprob / (float)depth;
    path_ppls[path_idx] = expf(avg_neg_lp);
}

// CUDA Kernel: Scales raw logprobs by temperature and normalizes over the B candidate tokens
__global__ void temperature_scale_normalize_kernel(
    const float* __restrict__ raw_logprobs, // [num_nodes, breadth]
    float* __restrict__ scaled_logprobs,    // [num_nodes, breadth]
    int num_nodes,
    int breadth,
    float temperature
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;

    float inv_temp = 1.0f / fmaxf(temperature, 1e-5f);
    int offset = node_idx * breadth;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int b = 0; b < breadth; ++b) {
        float val = raw_logprobs[offset + b] * inv_temp;
        if (val > max_val) max_val = val;
    }

    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int b = 0; b < breadth; ++b) {
        sum_exp += expf(raw_logprobs[offset + b] * inv_temp - max_val);
    }
    float log_sum_exp = max_val + logf(sum_exp);

    // Normalize
    for (int b = 0; b < breadth; ++b) {
        scaled_logprobs[offset + b] = (raw_logprobs[offset + b] * inv_temp) - log_sum_exp;
    }
}

// Host launcher function
torch::Tensor compute_path_perplexities_cuda(
    torch::Tensor candidate_logprobs, // [num_parent_nodes, breadth]
    torch::Tensor path_node_indices,  // [num_paths, depth]
    torch::Tensor path_child_indices  // [num_paths, depth]
) {
    auto cand_lp = candidate_logprobs.to(torch::kFloat32).contiguous();
    auto path_nodes = path_node_indices.to(torch::kInt32).contiguous();
    auto path_children = path_child_indices.to(torch::kInt32).contiguous();

    int num_paths = path_nodes.size(0);
    int depth = path_nodes.size(1);
    int breadth = cand_lp.size(1);

    auto path_ppls = torch::empty({num_paths}, cand_lp.options());

    int threads = 256;
    int blocks = (num_paths + threads - 1) / threads;

    compute_path_perplexity_kernel<<<blocks, threads>>>(
        cand_lp.data_ptr<float>(),
        path_nodes.data_ptr<int>(),
        path_children.data_ptr<int>(),
        path_ppls.data_ptr<float>(),
        num_paths,
        depth,
        breadth
    );

    return path_ppls;
}

torch::Tensor temperature_scale_normalize_cuda(
    torch::Tensor raw_logprobs, // [num_nodes, breadth]
    float temperature
) {
    auto raw_lp = raw_logprobs.to(torch::kFloat32).contiguous();
    int num_nodes = raw_lp.size(0);
    int breadth = raw_lp.size(1);

    auto scaled_logprobs = torch::empty_like(raw_lp);

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    temperature_scale_normalize_kernel<<<blocks, threads>>>(
        raw_lp.data_ptr<float>(),
        scaled_logprobs.data_ptr<float>(),
        num_nodes,
        breadth,
        temperature
    );

    return scaled_logprobs;
}

} // namespace cautious_decoding
