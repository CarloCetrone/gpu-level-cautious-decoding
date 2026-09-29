# vLLM GPU-Accelerated Cautious Tree Search Decoding (CTSD)

High-performance GPU-level plugin for **vLLM** implementing **Cautious Tree Search Decoding (CTSD)** from the research paper:

> *Cautious Tree Search Decoding: Perplexity-Guided Parallel Exploration for Language Model Generation* (Carlo Cetrone, 2026).

---

## ⚡ Architectural Overview & Performance Gains

In pure Python-level CTSD implementations, exploring $B^D$ candidate paths causes significant CPU-GPU synchronization overhead (host-device round-trips for dictionary sorting, candidate filtering, and math ops).

This GPU-level plugin eliminates host stalls by executing all tree scoring, candidate normalization, and perplexity evaluations directly in custom **CUDA C++ kernels / on-device GPU tensors**:

1. **CUDA Temperature Normalization Kernel (`temperature_scale_normalize_kernel`)**:
   $$p_\tau(c_j) = \frac{\exp(L_j / \tau)}{\sum_{m=1}^B \exp(L_m / \tau)}$$
   Scales and normalizes candidate log-probabilities strictly across the $B$ branch options directly in GPU registers, controlling sequence exploration without CPU intervention.

2. **CUDA Path Perplexity Reduction Kernel (`compute_path_perplexity_kernel`)**:
   $$\text{PPL}(p) = \exp\left(-\frac{1}{D} \sum_{t=1}^D \log p_\tau(y_t)\right)$$
   Evaluates all $B^D$ root-to-leaf paths in parallel using 2D CUDA thread grids with grid-stride loops, computing reductions directly in GPU shared memory.

3. **CUDA Tree Attention Mask Kernel (`build_tree_attention_mask_kernel`)**:
   Generates a 2D causal tree attention mask ($[N, N]$) directly on GPU VRAM, allowing all $B^D$ branches to be computed within a single forward pass without re-attending across disjoint candidate branches.

4. **Zero Host Synchronization Stalls**:
   Tree exploration, path accumulation, and argmin path selection remain fully resident on GPU VRAM until token commitment.

---

## 📦 Installation

### Option 1: Install with Native CUDA Compilation (Recommended for Colab / Linux GPU)
```bash
cd "vllm-gpu level cautious decoding"
pip install -e .
```
*Note: This automatically invokes `nvcc` and compiles `cautious_decoding_cuda`.*

### Option 2: Pure PyTorch Vectorized GPU Fallback
If `nvcc` or `CUDA_HOME` is not configured, the plugin automatically falls back to vectorized PyTorch GPU tensor math (`torch.logsumexp`, `torch.gather`), retaining high GPU throughput.

---

## 🚀 Quick Usage

```python
import vllm
import cautious_gpu # Automatically registers LLM.gpu_cautious_generate

# 1. Initialize vLLM with Automatic Prefix Caching
llm = vllm.LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=True,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
)

# 2. Execute GPU Cautious Decoding
result = llm.gpu_cautious_generate(
    prompt="Explain why cautious tree search decoding improves language model reasoning:",
    breadth=3,       # Breadth B (number of candidate tokens per branch)
    depth=3,         # Depth D (lookahead depth, evaluates B^D = 27 sequences)
    temperature=0.7, # Temperature scaling candidate distribution
    max_tokens=256,
    verbose=True,
)

print(result["generated_text"])
print(f"Throughput: {result['tokens_per_second']:.2f} tokens/sec")
```
