# SPDX-License-Identifier: Apache-2.0
"""
Tests for GPU-level tree scoring, temperature scaling, and tree attention masks.
"""

import torch
import pytest
from cautious_gpu.tree_kernel import (
    temperature_scale_normalize_gpu,
    compute_path_perplexities_gpu,
    build_tree_attention_mask_gpu,
)
from cautious_gpu.tree_attention import GPUTreeTopology


def test_temperature_scaling_and_normalization():
    # 2 nodes, 3 candidate tokens each
    raw_lps = torch.tensor([
        [-0.2, -1.5, -2.5],
        [-1.0, -1.1, -1.2]
    ], dtype=torch.float32)

    # Test high temperature: distribution flattens to ~1/3 (-1.0986)
    norm_high = temperature_scale_normalize_gpu(raw_lps, temperature=1000.0)
    assert torch.allclose(norm_high[0], torch.tensor([-1.0986, -1.0986, -1.0986]), atol=1e-2)

    # Test low temperature: distribution sharpens on top token
    norm_low = temperature_scale_normalize_gpu(raw_lps, temperature=0.1)
    assert norm_low[0, 0] > -0.01  # Near 0 logprob for greedy token
    assert norm_low[0, 1] < -10.0  # Heavily penalized


def test_gpu_path_perplexities():
    # Breadth 2, Depth 2 -> 4 paths
    # Nodes: 0 (root), 1 (child 0), 2 (child 1)
    candidate_lps = torch.tensor([
        [-0.5, -1.5], # node 0: choices 0, 1
        [-2.0, -2.5], # node 1: choices 0, 1
        [-0.1, -3.0], # node 2: choices 0, 1
    ], dtype=torch.float32)

    # 4 paths:
    # Path 0: node 0 choice 0, node 1 choice 0 -> (-0.5 + -2.0) = -2.5
    # Path 1: node 0 choice 0, node 1 choice 1 -> (-0.5 + -2.5) = -3.0
    # Path 2: node 0 choice 1, node 2 choice 0 -> (-1.5 + -0.1) = -1.6 (Best path!)
    # Path 3: node 0 choice 1, node 2 choice 1 -> (-1.5 + -3.0) = -4.5
    path_nodes = torch.tensor([
        [0, 1],
        [0, 1],
        [0, 2],
        [0, 2],
    ], dtype=torch.long)

    path_children = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.long)

    ppls = compute_path_perplexities_gpu(candidate_lps, path_nodes, path_children)
    assert ppls.shape == (4,)

    # Path 2 must have the lowest perplexity
    best_idx = torch.argmin(ppls).item()
    assert best_idx == 2


def test_tree_attention_mask():
    # Tree topology:
    # 0 (root) -> 1, 2
    # 1 -> 3
    parent_indices = torch.tensor([-1, 0, 0, 1], dtype=torch.long)
    mask = build_tree_attention_mask_gpu(parent_indices)

    # Self-attention must be True
    assert mask[0, 0] and mask[1, 1] and mask[2, 2] and mask[3, 3]

    # Node 1 and Node 2 attend to root 0
    assert mask[1, 0] and mask[2, 0]

    # Node 3 attends to 1 and 0, but NOT to sibling 2
    assert mask[3, 1] and mask[3, 0]
    assert not mask[3, 2]

    # Node 1 does NOT attend to child 3
    assert not mask[1, 3]


if __name__ == "__main__":
    test_temperature_scaling_and_normalization()
    test_gpu_path_perplexities()
    test_tree_attention_mask()
    print("All GPU tests passed!")
