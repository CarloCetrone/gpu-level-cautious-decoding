# SPDX-License-Identifier: Apache-2.0
"""
Tree Attention and static path layout generator for GPU-level execution as in Medusa.
"""

from __future__ import annotations

import itertools
import torch
from typing import Dict, List, Tuple


class GPUTreeTopology:
    """Precomputes static tensor indices for tree paths, attention masking, and position IDs.

    For a tree of breadth B and depth D:
    - Number of root-to-leaf paths: B^D
    - Number of candidate nodes (excluding root): sum_{d=1}^D B^d
    - Node indices and parent pointers are computed statically on GPU.
    """

    def __init__(self, breadth: int = 3, depth: int = 3, device: str = "cuda"):
        self.breadth = breadth
        self.depth = depth
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Total paths at depth D = B^D
        self.num_paths = breadth ** depth

        # Precompute path child indices [num_paths, depth]
        # Example for B=3, D=3: (0,0,0), (0,0,1), ..., (2,2,2)
        combinations = list(itertools.product(range(breadth), repeat=depth))
        self.path_child_indices = torch.tensor(
            combinations, dtype=torch.long, device=self.device
        )

        # Candidate nodes per depth level:
        # depth 1: B nodes (0 to B-1)
        # depth 2: B^2 nodes
        # ...
        self.num_nodes_per_depth = [breadth ** d for d in range(1, depth + 1)]
        self.num_candidate_nodes = sum(self.num_nodes_per_depth)

        # Depth offset of each level:
        self.depth_offsets = [0]
        for count in self.num_nodes_per_depth[:-1]:
            self.depth_offsets.append(self.depth_offsets[-1] + count)

        # Build candidate parent pointers [num_candidate_nodes]:
        # For depth 1 (indices 0..B-1): parent is -1 (root / prefix)
        # For depth > 1: node i has parent (i - B) // B
        cand_parents = []
        node_depths = []
        for d_idx, count in enumerate(self.num_nodes_per_depth):
            current_depth = d_idx + 1
            for _ in range(count):
                node_depths.append(current_depth)

        for i in range(self.num_candidate_nodes):
            if i < breadth:
                cand_parents.append(-1)
            else:
                cand_parents.append((i - breadth) // breadth)

        self.candidate_parent_indices = torch.tensor(
            cand_parents, dtype=torch.long, device=self.device
        )
        self.candidate_depths = torch.tensor(
            node_depths, dtype=torch.long, device=self.device
        )

    def get_path_node_indices(self) -> torch.Tensor:
        """Constructs [num_paths, depth] matrix giving the candidate node index for each step."""
        num_paths = self.num_paths
        depth = self.depth
        breadth = self.breadth

        path_nodes = torch.zeros((num_paths, depth), dtype=torch.long, device=self.device)
        child_indices = self.path_child_indices.cpu().numpy()

        for p_idx in range(num_paths):
            curr_node = 0
            for d in range(depth):
                child_choice = child_indices[p_idx, d]
                if d == 0:
                    curr_node = child_choice
                else:
                    curr_node = self.depth_offsets[d] + (curr_node - self.depth_offsets[d - 1]) * breadth + child_choice
                path_nodes[p_idx, d] = curr_node

        return path_nodes

    def get_tree_attention_mask(self) -> torch.Tensor:
        """Generates the 2D causal tree attention mask [N, N] for candidate nodes on GPU.

        mask[i, j] = True if node j is an ancestor of node i (or i == j), else False.
        """
        from cautious_gpu.tree_kernel import build_tree_attention_mask_gpu
        return build_tree_attention_mask_gpu(self.candidate_parent_indices)

    def get_4d_tree_attention_mask(
        self, prefix_len: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Generates the 4D causal tree attention mask for Transformer models as in Medusa.

        Shape: [1, 1, num_candidate_nodes, prefix_len + num_candidate_nodes]
        - For prefix columns [0, prefix_len): All candidate nodes attend (0.0).
        - For candidate columns [prefix_len, prefix_len + N):
          0.0 if node j is an ancestor of node i (or i == j), -inf otherwise.
        """
        n = self.num_candidate_nodes
        total_len = prefix_len + n

        # Start with full -inf
        mask_4d = torch.full(
            (1, 1, n, total_len),
            fill_value=float("-inf"),
            dtype=dtype,
            device=self.device,
        )

        # 1. All candidate nodes attend to all prefix tokens
        if prefix_len > 0:
            mask_4d[:, :, :, :prefix_len] = 0.0

        # 2. Candidate nodes attend only to themselves and their tree ancestors
        bool_tree_mask = self.get_tree_attention_mask()
        mask_4d[:, :, :, prefix_len:] = torch.where(
            bool_tree_mask,
            torch.tensor(0.0, dtype=dtype, device=self.device),
            torch.tensor(float("-inf"), dtype=dtype, device=self.device),
        )

        return mask_4d

    def get_tree_position_ids(self, prefix_len: int) -> torch.Tensor:
        """Generates tree position IDs reflecting depth in the tree for RoPE.

        Shape: [1, num_candidate_nodes]
        pos[0, i] = prefix_len + candidate_depth[i] - 1
        """
        return (prefix_len + self.candidate_depths - 1).unsqueeze(0)
