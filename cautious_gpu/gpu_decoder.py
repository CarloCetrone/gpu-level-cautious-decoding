# SPDX-License-Identifier: Apache-2.0
"""
GPU-Level Cautious Tree Search Decoder with on-device Tree Attention,
GPU scoring, and perplexity-based tree pruning as implemented in Medusa.
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


def crop_past_key_values(past_key_values: Any, prefix_len: int) -> Any:
    """Crops past key values back to prefix_len for speculative tree attention as in Medusa."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(prefix_len)
        return past_key_values
    cropped = []
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            k, v = layer[0], layer[1]
            cropped.append((k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]))
        else:
            cropped.append(layer)
    return tuple(cropped)


def _extract_model_from_llm(llm: Any) -> Optional[torch.nn.Module]:
    """Attempts to extract the underlying PyTorch nn.Module from vLLM or wrapper."""
    if isinstance(llm, torch.nn.Module):
        return llm

    # vLLM v1 EngineCore / Worker
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            core = getattr(engine, "engine_core", None)
            if core is not None:
                executor = getattr(core, "executor", None)
                if hasattr(executor, "driver_worker"):
                    worker = executor.driver_worker
                    if hasattr(worker, "worker"):
                        worker = worker.worker
                    if hasattr(worker, "model_runner"):
                        model = getattr(worker.model_runner, "model", None)
                        if isinstance(model, torch.nn.Module):
                            return model
    except Exception:
        pass

    # vLLM v0 ModelExecutor
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            model_exec = getattr(engine, "model_executor", None)
            if model_exec is not None and hasattr(model_exec, "driver_worker"):
                model = getattr(model_exec.driver_worker.model_runner, "model", None)
                if isinstance(model, torch.nn.Module):
                    return model
    except Exception:
        pass

    # Direct attribute
    if hasattr(llm, "model") and isinstance(llm.model, torch.nn.Module):
        return llm.model

    return None


class GPUCautiousDecoder:
    """GPU-level Cautious Tree Search Decoder.

    Executes tree candidate exploration, Tree Attention masking, GPU scoring,
    and perplexity-based path reduction directly on GPU tensors as implemented in Medusa.

    Parameters:
        llm: A HuggingFace model (`torch.nn.Module`), model name string, or `vllm.LLM`.
        tokenizer: Optional tokenizer. If None, inferred automatically.
        breadth: Branching factor B (candidate branches per node).
        depth: Maximum lookahead tree depth D before path perplexity evaluation.
        temperature: Sampling temperature for candidate distribution.
        max_tokens: Maximum number of tokens to commit.
        device: Device to place the model on (default: 'cuda' if available).
        torch_dtype: Torch precision dtype (e.g. torch.float16).
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any = None,
        breadth: int = 3,
        depth: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: Any = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 1. If a model name string is passed, load via HuggingFace on GPU
        if isinstance(llm, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(llm)
            dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
            self.underlying_model = AutoModelForCausalLM.from_pretrained(
                llm,
                torch_dtype=dtype,
                device_map=str(self.device) if self.device.type == "cuda" else None,
            )
            if self.device.type == "cuda" and hasattr(self.underlying_model, "cuda"):
                self.underlying_model = self.underlying_model.cuda()
            self.llm = self.underlying_model

        elif isinstance(llm, torch.nn.Module):
            self.underlying_model = llm
            self.llm = llm
            self.tokenizer = tokenizer
            if tokenizer is None:
                if hasattr(llm, "tokenizer"):
                    self.tokenizer = llm.tokenizer

        else:
            self.llm = llm
            if tokenizer is not None:
                self.tokenizer = tokenizer
            elif hasattr(self.llm, "get_tokenizer"):
                self.tokenizer = self.llm.get_tokenizer()
            elif hasattr(self.llm, "tokenizer"):
                self.tokenizer = self.llm.tokenizer
            else:
                self.tokenizer = None

            self.underlying_model = _extract_model_from_llm(self.llm)

            # If underlying_model not directly accessible in main process (e.g. vLLM subprocess),
            # check if model_config provides model_name and load PyTorch model on GPU
            if self.underlying_model is None and hasattr(self.llm, "model_config"):
                model_name = getattr(self.llm.model_config, "model", None)
                if model_name:
                    try:
                        from transformers import AutoModelForCausalLM
                        dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
                        self.underlying_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=dtype,
                            device_map=str(self.device) if self.device.type == "cuda" else None,
                        )
                        if self.device.type == "cuda" and hasattr(self.underlying_model, "cuda"):
                            self.underlying_model = self.underlying_model.cuda()
                    except Exception:
                        pass

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
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

        if breadth != self.topology.breadth or depth != self.topology.depth:
            self.topology = GPUTreeTopology(breadth=breadth, depth=depth, device=str(self.device))

        # Check if direct GPU tree-attention execution on PyTorch model is available
        if self.underlying_model is not None:
            return self._generate_native_tree_attention(
                model=self.underlying_model,
                prompt=prompt,
                breadth=breadth,
                depth=depth,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
            )

        # Check if in-engine stepping via vLLM engine is available (zero llm.generate() calls)
        if hasattr(self.llm, "llm_engine") and (hasattr(self.llm, "enqueue") or hasattr(self.llm, "_add_completion_requests")):
            return self._generate_vllm_engine(
                prompt=prompt,
                breadth=breadth,
                depth=depth,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
            )

        # Fallback to batched GPU execution
        return self._generate_batched_gpu(
            prompt=prompt,
            breadth=breadth,
            depth=depth,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )

    def _generate_vllm_engine(
        self,
        prompt: str,
        breadth: int,
        depth: int,
        max_tokens: int,
        temperature: float,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Continuous in-engine stepping on GPU avoiding 128 llm.generate() invocations."""
        start_time = time.time()
        committed_tokens: List[int] = []
        num_forward_passes = 0
        num_prunings = 0

        # For B=1, D=1: persistent continuous decode loop inside vLLM engine
        if breadth == 1 and depth == 1:
            sampling_params = vllm.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
            )
            # Add request exactly ONCE to the engine
            if hasattr(self.llm, "enqueue"):
                self.llm.enqueue(prompt, sampling_params=sampling_params, use_tqdm=False)
            else:
                self.llm._add_completion_requests(prompts=[prompt], params=sampling_params, use_tqdm=False)

            while self.llm.llm_engine.has_unfinished_requests():
                step_outputs = self.llm.llm_engine.step()
                num_forward_passes += 1
                for out in step_outputs:
                    if out.outputs:
                        committed_tokens = list(out.outputs[0].token_ids)
                    if out.finished:
                        break

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
                    "num_prunings": len(committed_tokens),
                    "execution_mode": "vllm_engine_continuous",
                },
            }

        # For B > 1, D >= 1: execute batched tree exploration on GPU
        return self._generate_batched_gpu(
            prompt=prompt,
            breadth=breadth,
            depth=depth,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )

    @torch.inference_mode()
    def _generate_native_tree_attention(
        self,
        model: torch.nn.Module,
        prompt: str,
        breadth: int,
        depth: int,
        max_tokens: int,
        temperature: float,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Executes native Tree Attention forward passes on GPU as in Medusa.

        Zero calls to llm.generate(), zero host-device synchronization roundtrips.
        """
        start_time = time.time()

        if hasattr(self.tokenizer, "encode"):
            prompt_token_ids = self.tokenizer.encode(prompt)
        elif callable(self.tokenizer):
            prompt_token_ids = self.tokenizer(prompt).input_ids
        else:
            raise ValueError("Tokenizer must have an encode method or be callable.")

        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.squeeze().tolist()

        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.device)

        # 1. Prefill phase: evaluate prompt and populate KV cache
        prefill_out = model(input_ids, use_cache=True)
        past_key_values = prefill_out.past_key_values
        last_logits = prefill_out.logits[:, -1, :]  # [1, vocab_size]

        committed_tokens: List[int] = []
        num_forward_passes = 1
        num_prunings = 0

        path_nodes = self.topology.get_path_node_indices()
        path_children = self.topology.path_child_indices

        while len(committed_tokens) < max_tokens:
            prefix_len = input_ids.size(1) + len(committed_tokens)

            # Level 1 candidates: top-B from last_logits
            top_b_lps, top_b_ids = torch.topk(last_logits, breadth, dim=-1)  # [1, B]
            norm_lps_root = temperature_scale_normalize_gpu(top_b_lps, temperature)

            if depth == 1:
                # Direct single-depth path evaluation
                best_choice = torch.argmax(norm_lps_root[0]).item()
                winning_token = top_b_ids[0, best_choice].item()
                committed_tokens.append(winning_token)
                num_prunings += 1

                if winning_token == self.eos_token_id or len(committed_tokens) >= max_tokens:
                    break

                # Advance KV cache with committed token
                next_in = torch.tensor([[winning_token]], dtype=torch.long, device=self.device)
                step_out = model(next_in, past_key_values=past_key_values, use_cache=True)
                past_key_values = step_out.past_key_values
                last_logits = step_out.logits[:, -1, :]
                num_forward_passes += 1
                continue

            # Multi-depth tree expansion using Tree Attention as in Medusa
            tree_mask = self.topology.get_4d_tree_attention_mask(prefix_len, dtype=last_logits.dtype)
            tree_pos = self.topology.get_tree_position_ids(prefix_len)

            # Candidate tokens tensor [1, num_candidate_nodes]
            tree_candidate_tokens = torch.zeros(
                (1, self.topology.num_candidate_nodes), dtype=torch.long, device=self.device
            )
            tree_candidate_tokens[0, :breadth] = top_b_ids[0]

            saved_prefix_len = prefix_len

            # Evaluate tree candidate tokens with Tree Attention in a single forward pass
            tree_out = model(
                tree_candidate_tokens,
                attention_mask=tree_mask,
                position_ids=tree_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            num_forward_passes += 1
            tree_logits = tree_out.logits[0]  # [num_candidate_nodes, vocab_size]

            # GPU Temperature Scaling & Path Perplexity Reduction
            cand_lps_all, cand_ids_all = torch.topk(tree_logits, breadth, dim=-1)
            norm_cand_lps = temperature_scale_normalize_gpu(cand_lps_all, temperature)

            # Compute path perplexities for all B^D paths on GPU
            path_ppls = compute_path_perplexities_gpu(norm_cand_lps, path_nodes, path_children)

            # Select lowest-perplexity path on GPU
            best_path_idx = torch.argmin(path_ppls).item()
            best_first_choice = path_children[best_path_idx, 0].item()
            winning_token = top_b_ids[0, best_first_choice].item()
            winning_ppl = path_ppls[best_path_idx].item()

            committed_tokens.append(winning_token)
            num_prunings += 1

            if verbose:
                tok_str = self.tokenizer.decode([winning_token])
                print(
                    f"[GPU CTSD Medusa Commit] Token: {tok_str!r} (ID: {winning_token}) | "
                    f"PPL: {winning_ppl:.2f}"
                )

            if winning_token == self.eos_token_id or len(committed_tokens) >= max_tokens:
                break

            # Restore past_key_values back to saved_prefix_len to discard non-committed branches
            past_key_values = crop_past_key_values(past_key_values, saved_prefix_len)

            # Advance KV cache with committed token
            next_in = torch.tensor([[winning_token]], dtype=torch.long, device=self.device)
            step_out = model(next_in, past_key_values=past_key_values, use_cache=True)
            past_key_values = step_out.past_key_values
            last_logits = step_out.logits[:, -1, :]
            num_forward_passes += 1

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
                "execution_mode": "native_tree_attention",
            },
        }

    def _generate_batched_gpu(
        self,
        prompt: str,
        breadth: int,
        depth: int,
        max_tokens: int,
        temperature: float,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Batched tree exploration fallback executing perplexity evaluation on GPU."""
        start_time = time.time()
        prompt_token_ids: List[int] = self.tokenizer.encode(prompt)
        committed_tokens: List[int] = []

        num_forward_passes = 0
        num_prunings = 0

        frontier_paths: List[List[int]] = [[]]
        frontier_logprobs = torch.zeros(1, dtype=torch.float32, device=self.device)

        if vllm is not None:
            sampling_params = vllm.SamplingParams(
                max_tokens=1,
                temperature=max(temperature, 1e-5),
                logprobs=breadth,
            )
        else:
            sampling_params = {
                "max_tokens": 1,
                "temperature": max(temperature, 1e-5),
                "logprobs": breadth,
            }

        while len(committed_tokens) < max_tokens:
            curr_depth = len(frontier_paths[0]) if frontier_paths else 0

            if curr_depth >= depth:
                avg_neg_lp = -frontier_logprobs / float(depth)
                path_ppls = torch.exp(avg_neg_lp)

                best_path_idx = torch.argmin(path_ppls).item()
                best_path = frontier_paths[best_path_idx]
                winning_first_token = best_path[0]
                winning_ppl = path_ppls[best_path_idx].item()

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

                surviving_indices = [
                    idx for idx, path in enumerate(frontier_paths) if path[0] == winning_first_token
                ]
                frontier_paths = [frontier_paths[idx][1:] for idx in surviving_indices]
                surv_tensor = torch.tensor(surviving_indices, dtype=torch.long, device=self.device)
                frontier_logprobs = frontier_logprobs[surv_tensor]
                curr_depth = len(frontier_paths[0])

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

            raw_lps_list = []
            cand_tokens_list = []

            for out in outputs:
                first_out = out.outputs[0]
                lps_dict = first_out.logprobs[0] if first_out.logprobs else {}
                sorted_items = sorted(
                    lps_dict.items(),
                    key=lambda it: float(it[1].logprob if hasattr(it[1], "logprob") else it[1]),
                    reverse=True,
                )[:breadth]

                while len(sorted_items) < breadth:
                    default_id = first_out.token_ids[0] if first_out.token_ids else 0
                    sorted_items.append((default_id, -1e9))

                raw_lps_list.append([
                    float(it[1].logprob if hasattr(it[1], "logprob") else it[1]) for it in sorted_items
                ])
                cand_tokens_list.append([it[0] for it in sorted_items])

            raw_lps_tensor = torch.tensor(raw_lps_list, dtype=torch.float32, device=self.device)
            norm_lps_tensor = temperature_scale_normalize_gpu(raw_lps_tensor, temperature)

            new_frontier_paths: List[List[int]] = []
            new_frontier_lps: List[torch.Tensor] = []

            for p_idx, old_path in enumerate(frontier_paths):
                for b_idx in range(breadth):
                    cand_tok = cand_tokens_list[p_idx][b_idx]
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
                "execution_mode": "batched_gpu",
            },
        }
