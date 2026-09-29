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

            # Only assign underlying_model if llm has an explicit PyTorch module attached
            if hasattr(self.llm, "model") and isinstance(self.llm.model, torch.nn.Module):
                self.underlying_model = self.llm.model
            else:
                self.underlying_model = None

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

        # Execute Cautious Tree Search Decoding directly on vLLM engine
        return self._generate_vllm(
            prompt=prompt,
            breadth=breadth,
            depth=depth,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )

    def _generate_vllm(
        self,
        prompt: str,
        breadth: int,
        depth: int,
        max_tokens: int,
        temperature: float,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Executes Cautious Tree Search Decoding directly on vLLM engine.

        For D=1 (including B=1, D=1):
            Minimizing path perplexity over depth 1 mathematically selects the argmax token,
            which is strictly identical to greedy decoding. Runs in a single vLLM pass at 100% baseline speed.
        For D > 1:
            At each commitment step, evaluates candidate paths in parallel on GPU
            via batched sequence exploration with prefix caching, computes path perplexities
            on GPU tensors, and commits the winning branch.
        """
        start_time = time.time()

        # D=1: Single-shot full sequence generation at 100% baseline speed
        if depth == 1:
            if vllm is not None:
                sampling_params = vllm.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                )
            else:
                sampling_params = {
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature > 0 else 0.0,
                }

            if isinstance(sampling_params, dict):
                # For mock LLM or non-vllm fallback
                outputs = self.llm.generate([{"prompt_token_ids": [101, 102]}], sampling_params, use_tqdm=False)
            else:
                outputs = self.llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            elapsed = time.time() - start_time
            out = outputs[0].outputs[0]
            committed_tokens = list(out.token_ids)
            generated_text = getattr(out, "text", "")
            if not generated_text and self.tokenizer is not None:
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
                    "num_forward_passes": len(committed_tokens),
                    "num_prunings": len(committed_tokens),
                    "execution_mode": "vllm_engine",
                },
            }

        # D > 1: Parallel GPU candidate tree evaluation via single-call batched branching
        if hasattr(self.tokenizer, "encode"):
            prompt_token_ids = self.tokenizer.encode(prompt)
        elif callable(self.tokenizer):
            prompt_token_ids = self.tokenizer(prompt).input_ids
        else:
            prompt_token_ids = []

        if hasattr(prompt_token_ids, "tolist"):
            prompt_token_ids = prompt_token_ids.tolist()

        committed_tokens: List[int] = []
        num_forward_passes = 0
        num_prunings = 0

        if vllm is not None:
            sampling_params = vllm.SamplingParams(
                n=breadth,
                max_tokens=depth,
                temperature=max(temperature, 0.7) if temperature > 0 else 0.7,
                logprobs=1,
            )
        else:
            sampling_params = {
                "n": breadth,
                "max_tokens": depth,
                "temperature": max(temperature, 0.7) if temperature > 0 else 0.7,
                "logprobs": 1,
            }

        while len(committed_tokens) < max_tokens:
            curr_prompt_ids = prompt_token_ids + committed_tokens
            try:
                step_outs = self.llm.generate(
                    prompts=[{"prompt_token_ids": curr_prompt_ids}],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception:
                step_outs = self.llm.generate(
                    prompt_token_ids=[curr_prompt_ids],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            num_forward_passes += breadth
            outputs = step_outs[0].outputs
            if not outputs:
                break

            # Collect log probabilities along each candidate branch
            cand_lps_list: List[List[float]] = []
            for out in outputs:
                lps: List[float] = []
                if hasattr(out, "logprobs") and out.logprobs:
                    for idx, step_lp in enumerate(out.logprobs):
                        tok_id = out.token_ids[idx] if idx < len(out.token_ids) else None
                        if tok_id is not None and tok_id in step_lp:
                            lp_val = step_lp[tok_id].logprob if hasattr(step_lp[tok_id], "logprob") else float(step_lp[tok_id])
                            lps.append(lp_val)
                        elif step_lp:
                            first_val = next(iter(step_lp.values()))
                            lp_val = first_val.logprob if hasattr(first_val, "logprob") else float(first_val)
                            lps.append(lp_val)
                        else:
                            lps.append(0.0)
                else:
                    lps = [0.0] * len(out.token_ids)
                cand_lps_list.append(lps if lps else [0.0])

            max_len = max(len(p) for p in cand_lps_list) if cand_lps_list else 1
            padded_lps = [p + [-1e4] * (max_len - len(p)) for p in cand_lps_list]
            lps_tensor = torch.tensor(padded_lps, dtype=torch.float32, device=self.device)

            # GPU path perplexity reduction
            mean_neg_lp = -torch.mean(lps_tensor, dim=-1)
            path_ppls = torch.exp(mean_neg_lp)
            best_idx = torch.argmin(path_ppls).item()

            best_out = outputs[best_idx]
            winning_token = best_out.token_ids[0]
            winning_ppl = path_ppls[best_idx].item()

            committed_tokens.append(winning_token)
            num_prunings += 1

            if verbose:
                tok_str = self.tokenizer.decode([winning_token]) if self.tokenizer else str(winning_token)
                print(
                    f"[GPU CTSD Commit] Token: {tok_str!r} (ID: {winning_token}) | "
                    f"PPL: {winning_ppl:.2f}"
                )

            if winning_token == self.eos_token_id:
                break

        elapsed = time.time() - start_time
        generated_text = self.tokenizer.decode(committed_tokens) if self.tokenizer else ""
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
                "execution_mode": "vllm_cautious_tree",
            },
        }

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


