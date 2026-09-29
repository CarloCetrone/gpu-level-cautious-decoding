# SPDX-License-Identifier: Apache-2.0
"""
Tree Attention and static path layout generator for GPU-level execution.
"""

from __future__ import annotations

import itertools
import torch
from typing import Dict, List, Tuple


class GPUTreeTopology:
    """Precomputes static tensor indices for tree paths and attention masking.

    For a tree of breadth B and depth D:
    - Number of root-to-leaf paths: B^D
    - Path indices are static permutations of child choices (0 to B-1).
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
            combinations, dtype=torch.int32, device=self.device
        )

        # Build dynamic node mapping buffers
        # At depth 1: B nodes (0 to B-1)
        # At depth 2: B^2 nodes (0 to B^2-1)
        # ...
        self.depth_offsets: List[int] = [0]
        curr_offset = 0
        for d in range(1, depth + 1):
            curr_offset += breadth ** (d - 1)
            self.depth_offsets.append(curr_offset)

    def get_path_node_indices(self) -> torch.Tensor:
        """Constructs [num_paths, depth] matrix giving the parent node index for each step."""
        num_paths = self.num_paths
        depth = self.depth
        breadth = self.breadth

        path_nodes = torch.zeros((num_paths, depth), dtype=torch.int32, device=self.device)
        child_indices = self.path_child_indices.cpu().numpy()

        for p_idx in range(num_paths):
            curr_parent = 0
            for d in range(depth):
                path_nodes[p_idx, d] = curr_parent
                child_choice = child_indices[p_idx, d]
                # In standard full tree, node index is updated as curr_parent * B + child_choice
                curr_parent = curr_parent * breadth + child_choice

        return path_nodes
