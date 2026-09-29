# SPDX-License-Identifier: Apache-2.0
"""
Benchmark & Inference Script: Pure GPU-Level Cautious Tree Search Decoding (Medusa-style Tree Attention)
vs Standard Greedy Autoregressive Baseline on GPU.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cautious_gpu import GPUCautiousDecoder


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Explain how cautious tree search decoding balances exploration and perplexity:"
    max_tokens = 128
    breadth = 3
    depth = 3
    temperature = 0.7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=================================================================")
    print(f"Loading Model: {model_name} on {device}...")
    print("=================================================================")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "cuda" and hasattr(model, "cuda"):
        model = model.cuda()

    # -------------------------------------------------------------
    # 1. Standard PyTorch Baseline (Greedy Autoregressive)
    # -------------------------------------------------------------
    print("\n[1/2] Running Standard PyTorch Baseline (Greedy)...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.inference_mode():
        baseline_outs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_time = time.time() - t0
    baseline_tokens = baseline_outs.shape[1] - input_ids.shape[1]
    baseline_tps = baseline_tokens / max(baseline_time, 1e-4)
    baseline_text = tokenizer.decode(baseline_outs[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"Baseline Time: {baseline_time:.2f}s | Speed: {baseline_tps:.2f} tokens/s")
    print(f"Generated ({baseline_tokens} tokens):\n{baseline_text}\n")

    # -------------------------------------------------------------
    # 2. Native GPU Cautious Tree Search Decoding (Medusa-style Tree Attention)
    # -------------------------------------------------------------
    print(f"\n[2/2] Running Native GPU Tree Attention CTSD (B={breadth}, D={depth}, B^D={breadth**depth} paths)...")
    decoder = GPUCautiousDecoder(
        llm=model,
        tokenizer=tokenizer,
        breadth=breadth,
        depth=depth,
        temperature=temperature,
        max_tokens=max_tokens,
        device=device,
    )
    gpu_result = decoder.generate(prompt=prompt, verbose=False)

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
    print(f"Baseline Throughput:     {baseline_tps:.2f} tokens/s")
    print(f"GPU CTSD Throughput:     {gpu_tps:.2f} tokens/s (B={breadth}, D={depth})")
    print(f"Speed Ratio:             {gpu_tps / baseline_tps:.1%}")
    print(f"Forward passes executed: {gpu_result['stats']['num_forward_passes']}")
    print(f"Tree pruning decisions:  {gpu_result['stats']['num_prunings']}")
    print(f"Execution mode:          {gpu_result['stats'].get('execution_mode', 'gpu')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
