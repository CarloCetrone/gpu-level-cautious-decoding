# SPDX-License-Identifier: Apache-2.0
"""
High-Performance GPU-Level Cautious Tree Search Decoder with on-device scoring,
vectorized perplexity reduction, and adaptive tree exploration.
"""

from __future__ import annotations

import time
import torch
import math
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


def _extract_topk_from_vllm_output(
    out: Any, breadth: int
) -> Tuple[List[int], List[float], float]:
    """Fast extraction of top-K candidate tokens and logprobs from vLLM output.

    Returns:
        candidate_token_ids: list of int
        candidate_logprobs: list of float
        top1_probability: probability of the top-1 candidate (0.0 to 1.0)
    """
    first_out = out.outputs[0]
    logprobs_dict = first_out.logprobs[0] if first_out.logprobs else None

    if not logprobs_dict:
        # Fallback if logprobs weren't returned
        top_token = first_out.token_ids[0] if first_out.token_ids else 0
        return [top_token] * breadth, [0.0] + [-1e9] * (breadth - 1), 1.0

    # Extract items: vLLM returns either float or Logprob object with .logprob
    items = [
        (tok_id, float(val.logprob if hasattr(val, "logprob") else val))
        for tok_id, val in logprobs_dict.items()
    ]
    # Fast sort top-B
    items.sort(key=lambda x: x[1], reverse=True)
    top_items = items[:breadth]

    while len(top_items) < breadth:
        default_id = first_out.token_ids[0] if first_out.token_ids else 0
        top_items.append((default_id, -1e9))

    cand_ids = [it[0] for it in top_items]
    cand_lps = [it[1] for it in top_items]
    top1_prob = math.exp(min(0.0, cand_lps[0]))

    return cand_ids, cand_lps, top1_prob


class GPUCautiousDecoder:
    """High-performance GPU-level Cautious Tree Search Decoder.

    Executes tree scoring, temperature normalization, and path perplexity reduction
    directly on GPU VRAM with minimal host overhead and adaptive tree exploration.

    Parameters:
        llm: An instance of `vllm.LLM`.
        breadth: Branching factor B (candidate branches per node).
        depth: Maximum lookahead tree depth D before path perplexity evaluation.
        temperature: Sampling temperature for candidate exploration.
        max_tokens: Maximum number of tokens to commit.
        adaptive_cautious: If True, uses confidence-guided cautious exploration
            (greedy shortcut on confident tokens, tree search on uncertain tokens),
            bringing execution speed to near-baseline vLLM throughput.
        confidence_threshold: Probability threshold above which a token is deemed
            confident enough to bypass branching (default: 0.85).
        commit_lookahead: If True, commits the verified tokens along the winning path
            p*, amortizing the tree search cost across multiple tokens per rollout.
    """

    def __init__(
        self,
        llm: Any,
        breadth: int = 3,
        depth: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
        adaptive_cautious: bool = True,
        confidence_threshold: float = 0.85,
        commit_lookahead: bool = True,
    ):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.adaptive_cautious = adaptive_cautious
        self.confidence_threshold = confidence_threshold
        self.commit_lookahead = commit_lookahead

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
        adaptive_cautious: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
        commit_lookahead: Optional[bool] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Executes GPU-accelerated Cautious Tree Search Decoding."""
        breadth = breadth or self.breadth
        depth = depth or self.depth
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        adaptive_cautious = (
            adaptive_cautious if adaptive_cautious is not None else self.adaptive_cautious
        )
        confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )
        commit_lookahead = (
            commit_lookahead if commit_lookahead is not None else self.commit_lookahead
        )

        if vllm is None and not hasattr(self.llm, "generate"):
            raise RuntimeError(
                "vLLM is required to execute GPUCautiousDecoder.generate(). Please install vllm."
            )

        start_time = time.time()
        prompt_token_ids: List[int] = self.tokenizer.encode(prompt)
        committed_tokens: List[int] = []

        num_forward_passes = 0
        num_prunings = 0
        num_greedy_shortcuts = 0
        num_tree_explorations = 0

        # State for active tree frontier
        frontier_paths: List[List[int]] = [[]]
        frontier_logprobs = torch.zeros(1, dtype=torch.float32, device=self.device)

        if vllm is not None:
            sampling_params = vllm.SamplingParams(
                max_tokens=1,
                temperature=max(temperature, 1e-5),
                logprobs=breadth,
            )
        else:
            # Fallback dict for mock LLMs in unit tests
            sampling_params = {
                "max_tokens": 1,
                "temperature": max(temperature, 1e-5),
                "logprobs": breadth,
            }

        while len(committed_tokens) < max_tokens:
            curr_depth = len(frontier_paths[0]) if frontier_paths else 0

            # -------------------------------------------------------------
            # 1. Pruning and Path Commitment (When depth D is reached)
            # -------------------------------------------------------------
            if curr_depth >= depth:
                # Vectorized Path Perplexity Reduction on GPU
                avg_neg_lp = -frontier_logprobs / float(depth)
                path_ppls = torch.exp(avg_neg_lp)
                best_path_idx = torch.argmin(path_ppls).item()
                best_path = frontier_paths[best_path_idx]
                winning_ppl = path_ppls[best_path_idx].item()
                num_prunings += 1

                if commit_lookahead:
                    # Multi-token commitment: Commit verified tokens along the winning path
                    tokens_to_commit = best_path[: max_tokens - len(committed_tokens)]
                    for tok in tokens_to_commit:
                        committed_tokens.append(tok)
                        if verbose:
                            tok_str = self.tokenizer.decode([tok])
                            print(
                                f"[GPU CTSD Commit (Lookahead)] Token: {tok_str!r} (ID: {tok}) | "
                                f"PPL: {winning_ppl:.2f}"
                            )
                        if tok == self.eos_token_id or len(committed_tokens) >= max_tokens:
                            break

                    if (
                        committed_tokens
                        and committed_tokens[-1] == self.eos_token_id
                        or len(committed_tokens) >= max_tokens
                    ):
                        break

                    # Reset tree frontier for next exploration
                    frontier_paths = [[]]
                    frontier_logprobs = torch.zeros(1, dtype=torch.float32, device=self.device)
                    curr_depth = 0
                else:
                    # Single-token commitment: Retain surviving branches starting with winning token
                    winning_first_token = best_path[0]
                    committed_tokens.append(winning_first_token)
                    if verbose:
                        tok_str = self.tokenizer.decode([winning_first_token])
                        print(
                            f"[GPU CTSD Commit] Token: {tok_str!r} (ID: {winning_first_token}) | "
                            f"PPL: {winning_ppl:.2f}"
                        )
                    if winning_first_token == self.eos_token_id or len(committed_tokens) >= max_tokens:
                        break

                    surviving_indices = [
                        idx for idx, path in enumerate(frontier_paths) if path[0] == winning_first_token
                    ]
                    frontier_paths = [frontier_paths[idx][1:] for idx in surviving_indices]
                    surv_tensor = torch.tensor(surviving_indices, dtype=torch.long, device=self.device)
                    frontier_logprobs = frontier_logprobs[surv_tensor]
                    curr_depth = len(frontier_paths[0])

            # -------------------------------------------------------------
            # 2. Batched Forward Pass Over Current Frontier
            # -------------------------------------------------------------
            batch_prompts = [
                {"prompt_token_ids": prompt_token_ids + committed_tokens + path}
                for path in frontier_paths
            ]

            outputs = self.llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            num_forward_passes += len(batch_prompts)

            # -------------------------------------------------------------
            # 3. Check for Adaptive Greedy Shortcut (at tree root)
            # -------------------------------------------------------------
            if adaptive_cautious and curr_depth == 0 and len(frontier_paths) == 1:
                cand_ids, cand_lps, top1_prob = _extract_topk_from_vllm_output(outputs[0], breadth)
                if top1_prob >= confidence_threshold:
                    # Model is highly confident: bypass branching and commit immediately!
                    greedy_tok = cand_ids[0]
                    committed_tokens.append(greedy_tok)
                    num_greedy_shortcuts += 1

                    if verbose:
                        tok_str = self.tokenizer.decode([greedy_tok])
                        print(
                            f"[GPU CTSD Fast Commit] Token: {tok_str!r} (ID: {greedy_tok}) | "
                            f"Confidence: {top1_prob:.1%}"
                        )

                    if greedy_tok == self.eos_token_id or len(committed_tokens) >= max_tokens:
                        break
                    continue

            # -------------------------------------------------------------
            # 4. Tree Frontier Expansion with GPU Temperature Normalization
            # -------------------------------------------------------------
            num_tree_explorations += 1
            raw_lps_list: List[List[float]] = []
            candidate_tokens_list: List[List[int]] = []

            for out in outputs:
                cand_ids, cand_lps, _ = _extract_topk_from_vllm_output(out, breadth)
                candidate_tokens_list.append(cand_ids)
                raw_lps_list.append(cand_lps)

            # Upload to GPU and normalize across candidate dimension
            raw_lps_tensor = torch.tensor(raw_lps_list, dtype=torch.float32, device=self.device)
            norm_lps_tensor = temperature_scale_normalize_gpu(raw_lps_tensor, temperature)

            # Expand frontier paths by breadth B
            new_frontier_paths: List[List[int]] = []
            new_frontier_lps: List[torch.Tensor] = []

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
                "num_greedy_shortcuts": num_greedy_shortcuts,
                "num_tree_explorations": num_tree_explorations,
                "adaptive_cautious": adaptive_cautious,
                "commit_lookahead": commit_lookahead,
            },
        }
