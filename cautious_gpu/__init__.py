# SPDX-License-Identifier: Apache-2.0
"""
GPU-Level Cautious Tree Search Decoding.
"""

from typing import Any
from cautious_gpu.gpu_decoder import GPUCautiousDecoder
from cautious_gpu.tree_attention import GPUTreeTopology
from cautious_gpu.tree_kernel import (
    compute_path_perplexities_gpu,
    temperature_scale_normalize_gpu,
    build_tree_attention_mask_gpu,
)
from cautious_gpu.plugin import register


def load_model(
    model_name: str,
    torch_dtype: Any = None,
    device: str = "cuda",
    breadth: int = 3,
    depth: int = 3,
    temperature: float = 0.7,
) -> GPUCautiousDecoder:
    """Convenience helper to load a HuggingFace causal LM on GPU ready for Medusa-style Cautious Tree Decoding."""
    import torch
    dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    return GPUCautiousDecoder(
        llm=model_name,
        breadth=breadth,
        depth=depth,
        temperature=temperature,
        device=device,
        torch_dtype=dtype,
    )


__version__ = "0.2.0"
__all__ = [
    "GPUCautiousDecoder",
    "GPUTreeTopology",
    "compute_path_perplexities_gpu",
    "temperature_scale_normalize_gpu",
    "build_tree_attention_mask_gpu",
    "load_model",
    "register",
]
