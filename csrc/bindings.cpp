// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Carlo Cetrone. All rights reserved.
// PyTorch C++ / CUDA Bindings for Cautious Tree Search Decoding.

#include <torch/extension.h>

namespace cautious_decoding {

// Declarations of CUDA launchers
torch::Tensor compute_path_perplexities_cuda(
    torch::Tensor candidate_logprobs,
    torch::Tensor path_node_indices,
    torch::Tensor path_child_indices
);

torch::Tensor temperature_scale_normalize_cuda(
    torch::Tensor raw_logprobs,
    float temperature
);

torch::Tensor build_tree_attention_mask_cuda(
    torch::Tensor parent_indices
);

} // namespace cautious_decoding

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Cautious Tree Search Decoding C++/CUDA Extension";
    m.def(
        "compute_path_perplexities",
        &cautious_decoding::compute_path_perplexities_cuda,
        "Computes path perplexities across all tree paths in parallel on GPU."
    );
    m.def(
        "temperature_scale_normalize",
        &cautious_decoding::temperature_scale_normalize_cuda,
        "Normalizes candidate log-probabilities with temperature on GPU."
    );
    m.def(
        "build_tree_attention_mask",
        &cautious_decoding::build_tree_attention_mask_cuda,
        "Constructs a 2D causal tree attention mask on GPU."
    );
}
