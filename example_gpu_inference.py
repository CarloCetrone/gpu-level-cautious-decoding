# SPDX-License-Identifier: Apache-2.0
"""
Benchmark & Inference Script: GPU-Level Cautious Tree Search Decoding vs Standard vLLM Baseline.
"""

import time
import torch
import vllm
import cautious_gpu


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Explain how cautious tree search decoding balances exploration and perplexity:"
    max_tokens = 128
    breadth = 3
    depth = 3
    temperature = 0.7

    print("=================================================================")
    print(f"Loading Model: {model_name} with Prefix Caching...")
    print("=================================================================")
    llm = vllm.LLM(
        model=model_name,
        enable_prefix_caching=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )

    # -------------------------------------------------------------
    # 1. Standard vLLM Baseline
    # -------------------------------------------------------------
    print("\n[1/2] Running Standard vLLM Baseline (Greedy)...")
    baseline_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    t0 = time.time()
    baseline_outs = llm.generate(prompts=[prompt], sampling_params=baseline_params, use_tqdm=False)
    baseline_time = time.time() - t0

    baseline_tokens = len(baseline_outs[0].outputs[0].token_ids)
    baseline_tps = baseline_tokens / max(baseline_time, 1e-4)
    baseline_text = baseline_outs[0].outputs[0].text.strip()

    print(f"Baseline Time: {baseline_time:.2f}s | Speed: {baseline_tps:.2f} tokens/s")
    print(f"Generated ({baseline_tokens} tokens):\n{baseline_text}\n")

    # -------------------------------------------------------------
    # 2. Fast GPU Cautious Tree Search Decoding (Adaptive & Lookahead)
    # -------------------------------------------------------------
    print(f"\n[2/2] Running Fast GPU Cautious Decoding (Adaptive, B={breadth}, D={depth})...")
    gpu_result = llm.gpu_cautious_generate(
        prompt=prompt,
        breadth=breadth,
        depth=depth,
        temperature=temperature,
        max_tokens=max_tokens,
        adaptive_cautious=True,
        confidence_threshold=0.85,
        commit_lookahead=True,
        verbose=False,
    )

    gpu_tokens = gpu_result["num_committed_tokens"]
    gpu_tps = gpu_result["tokens_per_second"]
    gpu_time = gpu_result["elapsed_time_sec"]
    gpu_text = gpu_result["generated_text"].strip()

    print(f"GPU CTSD Time: {gpu_time:.2f}s | Speed: {gpu_tps:.2f} tokens/s")
    print(f"Generated ({gpu_tokens} tokens):\n{gpu_text}\n")

    # -------------------------------------------------------------
    # Summary Comparison
    # -------------------------------------------------------------
    print("=" * 65)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"Baseline Throughput:       {baseline_tps:.2f} tokens/s")
    print(f"Fast GPU CTSD Throughput:  {gpu_tps:.2f} tokens/s (Ratio: {gpu_tps / max(baseline_tps, 1e-4):.1%})")
    print(f"Forward passes executed:   {gpu_result['stats']['num_forward_passes']}")
    print(f"Greedy shortcuts (fast):   {gpu_result['stats']['num_greedy_shortcuts']}")
    print(f"Tree explorations:         {gpu_result['stats']['num_tree_explorations']}")
    print(f"Tree pruning decisions:    {gpu_result['stats']['num_prunings']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
