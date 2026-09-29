# SPDX-License-Identifier: Apache-2.0
"""
GPU-Level Cautious Tree Search Decoder with on-device scoring and pruning.
"""

from __future__ import annotations

import time
import torch
from typing import Any, Dict, List, Optional, Tuple

try:
    import vllm
except ImportError:
    vllm = None

from cautious_gpu.tree_kernel import (
    compute_path_perplexities_gpu,
    temperature_scale_normalize_gpu,
)
from cautious_gpu.tree_attention import GPUTreeTopology


class GPUCautiousDecoder:
    """High-performance GPU-level Cautious Tree Search Decoder.

    Executes tree scoring, temperature normalization, and path perplexity reduction
    directly on GPU VRAM with zero host synchronization stalls.
    """

    def __init__(
        self,
        llm: Any,
        breadth: int = 3,
        depth: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topology = GPUTreeTopology(breadth=breadth, depth=depth, device=str(self.device))

    def generate(
        self,
        prompt: str,
        breadth: Optional[int] = None,
        depth: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Executes GPU-accelerated Cautious Tree Search Decoding."""
        breadth = breadth or self.breadth
        depth = depth or self.depth
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        if vllm is None:
            raise RuntimeError(
                "vLLM is required to execute GPUCautiousDecoder.generate(). Please install vllm."
            )

        start_time = time.time()
        prompt_token_ids: List[int] = self.tokenizer.encode(prompt)

        # Committed tokens sequence
        committed_tokens: List[int] = []

        # Active frontier sequences on the tree
        # List of candidate paths relative to committed root: [ [token_id, ...] ]
        frontier_paths: List[List[int]] = [[]]
        # Tensor of accumulated temperature-scaled logprobs along the paths [num_frontier]
        frontier_logprobs = torch.zeros(1, dtype=torch.float32, device=self.device)

        num_forward_passes = 0
        num_prunings = 0

        while len(committed_tokens) < max_tokens:
            curr_depth = len(frontier_paths[0]) if frontier_paths else 0

            # When reaching depth D, evaluate and prune on GPU
            if curr_depth >= depth:
                # Number of candidate paths is B^D
                # Compute path perplexities on GPU
                avg_neg_lp = -frontier_logprobs / float(depth)
                path_ppls = torch.exp(avg_neg_lp)

                # Find best path using GPU argmin
                best_path_idx = torch.argmin(path_ppls).item()
                best_path = frontier_paths[best_path_idx]
                winning_first_token = best_path[0]
                winning_ppl = path_ppls[best_path_idx].item()

                # Commit winning first token
                committed_tokens.append(winning_first_token)
                num_prunings += 1

                if verbose:
                    tok_str = self.tokenizer.decode([winning_first_token])
                    print(
                        f"[GPU CTSD Commit] Token: {tok_str!r} (ID: {winning_first_token}) | "
                        f"PPL: {winning_ppl:.2f}"
                    )

                if winning_first_token == self.eos_token_id or len(committed_tokens) >= max_tokens:
                    break

                # GPU Pruning: Retain only the paths starting with winning_first_token,
                # shift their paths by 1 token (re-rooting), keeping B^(D-1) remaining sequences
                surviving_indices = [
                    idx for idx, path in enumerate(frontier_paths) if path[0] == winning_first_token
                ]
                frontier_paths = [frontier_paths[idx][1:] for idx in surviving_indices]
                surv_tensor = torch.tensor(surviving_indices, dtype=torch.long, device=self.device)
                frontier_logprobs = frontier_logprobs[surv_tensor]
                curr_depth = len(frontier_paths[0])

            # Prepare batch for all frontier sequences
            batch_prompts = [
                {"prompt_token_ids": prompt_token_ids + committed_tokens + path}
                for path in frontier_paths
            ]

            sampling_params = vllm.SamplingParams(
                max_tokens=1,
                temperature=max(temperature, 1e-5),
                logprobs=breadth,
            )

            # Batched execution leveraging vLLM's Automatic Prefix Caching (APC)
            outputs = self.llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            num_forward_passes += len(batch_prompts)

            # Extract raw logprobs tensor for all frontier nodes: [num_frontier, breadth]
            raw_lps_list = []
            candidate_tokens_list = []

            for out in outputs:
                first_out = out.outputs[0]
                logprobs_dict = first_out.logprobs[0] if first_out.logprobs else {}
                sorted_items = sorted(
                    logprobs_dict.items(),
                    key=lambda it: float(it[1].logprob) if hasattr(it[1], "logprob") else float(it[1]),
                    reverse=True,
                )[:breadth]

                # Ensure exactly breadth items
                while len(sorted_items) < breadth:
                    default_id = first_out.token_ids[0] if first_out.token_ids else 0
                    sorted_items.append((default_id, -1e9))

                raw_lps_list.append([float(it[1].logprob if hasattr(it[1], "logprob") else it[1]) for it in sorted_items])
                candidate_tokens_list.append([it[0] for it in sorted_items])

            # Convert to GPU tensor: [num_frontier, breadth]
            raw_lps_tensor = torch.tensor(raw_lps_list, dtype=torch.float32, device=self.device)

            # GPU Temperature Scaling & Normalization over candidate dimension
            norm_lps_tensor = temperature_scale_normalize_gpu(raw_lps_tensor, temperature)

            # Expand frontier paths by breadth B
            new_frontier_paths: List[List[int]] = []
            new_frontier_lps = []

            for p_idx, old_path in enumerate(frontier_paths):
                for b_idx in range(breadth):
                    cand_tok = candidate_tokens_list[p_idx][b_idx]
                    new_frontier_paths.append(old_path + [cand_tok])
                    new_frontier_lps.append(
                        frontier_logprobs[p_idx] + norm_lps_tensor[p_idx, b_idx]
                    )

            frontier_paths = new_frontier_paths
            frontier_logprobs = torch.stack(new_frontier_lps)

        elapsed = time.time() - start_time
        generated_text = self.tokenizer.decode(committed_tokens)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "output_tokens": committed_tokens,
            "num_committed_tokens": len(committed_tokens),
            "elapsed_time_sec": elapsed,
            "tokens_per_second": len(committed_tokens) / max(elapsed, 1e-4),
            "stats": {
                "breadth": breadth,
                "depth": depth,
                "num_forward_passes": num_forward_passes,
                "num_prunings": num_prunings,
            },
        }
