# SPDX-License-Identifier: Apache-2.0
"""
GPU Kernel interface for Cautious Tree Search Decoding.
Loads compiled CUDA C++ extension or falls back to optimized PyTorch tensor operations.
"""

from __future__ import annotations

import torch
from typing import Tuple

_CUDA_EXT = None

# 1. Try importing pre-compiled extension
try:
    import cautious_decoding_cuda as _CUDA_EXT
except ImportError:
    pass

# 2. If not pre-compiled, attempt on-demand JIT compilation via torch.utils.cpp_extension
if _CUDA_EXT is None and torch.cuda.is_available():
    try:
        import os
        from torch.utils.cpp_extension import load

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.abspath(os.path.join(current_dir, "..", "csrc"))
        sources = [
            os.path.join(csrc_dir, "bindings.cpp"),
            os.path.join(csrc_dir, "tree_scoring.cu"),
            os.path.join(csrc_dir, "tree_attention.cu"),
        ]
        if all(os.path.exists(s) for s in sources):
            _CUDA_EXT = load(
                name="cautious_decoding_cuda",
                sources=sources,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
    except Exception:
        # Fall back to PyTorch GPU vectorized operations
        _CUDA_EXT = None


def temperature_scale_normalize_gpu(
    raw_logprobs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Scales raw logprobs by temperature and normalizes over candidate dimension.

    Args:
        raw_logprobs: [num_nodes, breadth] tensor of log-probabilities
        temperature: scalar float > 0
    Returns:
        [num_nodes, breadth] normalized log-probabilities
    """
    if _CUDA_EXT is not None and raw_logprobs.is_cuda:
        return _CUDA_EXT.temperature_scale_normalize(raw_logprobs, float(temperature))

    # Fast PyTorch vectorized GPU implementation
    eff_temp = max(temperature, 1e-5)
    scaled = raw_logprobs / eff_temp
    log_sum_exp = torch.logsumexp(scaled, dim=-1, keepdim=True)
    return scaled - log_sum_exp


def compute_path_perplexities_gpu(
    candidate_logprobs: torch.Tensor,  # [num_nodes, breadth]
    path_node_indices: torch.Tensor,  # [num_paths, depth]
    path_child_indices: torch.Tensor,  # [num_paths, depth]
) -> torch.Tensor:
    """Computes perplexity for all root-to-leaf paths entirely on GPU in parallel.

    PPL(p) = exp(-1/depth * sum_{t=1}^depth log p(y_t))
    """
    if _CUDA_EXT is not None and candidate_logprobs.is_cuda:
        return _CUDA_EXT.compute_path_perplexities(
            candidate_logprobs, path_node_indices, path_child_indices
        )

    # Fast PyTorch vectorized GPU implementation:
    # Gather logprobs for each node and child along the paths
    depth = path_node_indices.size(1)
    if path_node_indices.device != candidate_logprobs.device:
        path_node_indices = path_node_indices.to(candidate_logprobs.device)
    if path_child_indices.device != candidate_logprobs.device:
        path_child_indices = path_child_indices.to(candidate_logprobs.device)

    # [num_paths, depth]
    gathered_logprobs = candidate_logprobs[path_node_indices, path_child_indices]
    sum_logprobs = gathered_logprobs.sum(dim=1)
    avg_neg_lp = -sum_logprobs / depth
    return torch.exp(avg_neg_lp)


def build_tree_attention_mask_gpu(parent_indices: torch.Tensor) -> torch.Tensor:
    """Constructs the 2D causal tree attention mask [N, N] on GPU.

    mask[i, j] = True if j is an ancestor of i (or i == j), else False.
    """
    if _CUDA_EXT is not None and parent_indices.is_cuda:
        return _CUDA_EXT.build_tree_attention_mask(parent_indices)

    # PyTorch vectorized GPU implementation
    num_tokens = parent_indices.size(0)
    device = parent_indices.device
    mask = torch.eye(num_tokens, dtype=torch.bool, device=device)

    # Iterative ancestor traversal
    curr_parents = parent_indices.clone()
    for _ in range(num_tokens):
        valid = curr_parents >= 0
        if not valid.any():
            break
        # Query tokens i attend to curr_parents[i]
        valid_idx = torch.where(valid)[0]
        p_idx = curr_parents[valid_idx]
        mask[valid_idx, p_idx] = True
        next_parents = torch.full_like(curr_parents, -1)
        next_parents[valid_idx] = parent_indices[p_idx]
        curr_parents = next_parents

    return mask
