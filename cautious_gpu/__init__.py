# SPDX-License-Identifier: Apache-2.0
"""
GPU-Level Cautious Tree Search Decoding for vLLM.
"""

from cautious_gpu.gpu_decoder import GPUCautiousDecoder
from cautious_gpu.tree_attention import GPUTreeTopology
from cautious_gpu.tree_kernel import (
    compute_path_perplexities_gpu,
    temperature_scale_normalize_gpu,
    build_tree_attention_mask_gpu,
)
from cautious_gpu.plugin import register

__version__ = "0.2.0"
__all__ = [
    "GPUCautiousDecoder",
    "GPUTreeTopology",
    "compute_path_perplexities_gpu",
    "temperature_scale_normalize_gpu",
    "build_tree_attention_mask_gpu",
    "register",
]
