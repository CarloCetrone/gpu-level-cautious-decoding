# SPDX-License-Identifier: Apache-2.0
"""
Plugin registration for GPU-level Cautious Tree Search Decoding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def register():
    """vLLM plugin discovery entrypoint."""
    try:
        import vllm
        from cautious_gpu.gpu_decoder import GPUCautiousDecoder

        def gpu_cautious_generate(
            self,
            prompt: str,
            breadth: int = 3,
            depth: int = 3,
            max_tokens: int = 512,
            temperature: float = 0.7,
            verbose: bool = False,
        ) -> Dict[str, Any]:
            """GPU-level Cautious Tree Search Decoding.

            Parameters:
                prompt: Input prompt string.
                breadth: Branching factor B (candidate branches per node).
                depth: Lookahead exploration depth D.
                max_tokens: Maximum tokens to generate.
                temperature: Sampling temperature for candidate distribution.
                verbose: If True, logs step-by-step commitment info.
            """
            decoder = GPUCautiousDecoder(
                llm=self,
                breadth=breadth,
                depth=depth,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return decoder.generate(
                prompt=prompt,
                verbose=verbose,
            )

        vllm.LLM.gpu_cautious_generate = gpu_cautious_generate
        logger.info("[GPUCautiousDecoding] Registered `LLM.gpu_cautious_generate`.")
    except Exception as e:
        logger.warning(f"[GPUCautiousDecoding] Registration failed: {e}")
