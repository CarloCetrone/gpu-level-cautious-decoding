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
            adaptive_cautious: bool = True,
            confidence_threshold: float = 0.85,
            commit_lookahead: bool = True,
            verbose: bool = False,
        ) -> Dict[str, Any]:
            """GPU-accelerated Cautious Tree Search Decoding.

            Parameters:
                prompt: Input prompt string.
                breadth: Branching factor B (candidate branches per node).
                depth: Lookahead exploration depth D.
                max_tokens: Maximum tokens to generate.
                temperature: Sampling temperature.
                adaptive_cautious: If True, uses uncertainty-guided exploration,
                    committing confident tokens greedily to achieve near-baseline speed.
                confidence_threshold: Probability threshold for confident tokens (default 0.85).
                commit_lookahead: If True, commits verified tokens along the winning path.
                verbose: If True, logs step-by-step commitment info.
            """
            decoder = GPUCautiousDecoder(
                llm=self,
                breadth=breadth,
                depth=depth,
                temperature=temperature,
                max_tokens=max_tokens,
                adaptive_cautious=adaptive_cautious,
                confidence_threshold=confidence_threshold,
                commit_lookahead=commit_lookahead,
            )
            return decoder.generate(
                prompt=prompt,
                verbose=verbose,
            )

        vllm.LLM.gpu_cautious_generate = gpu_cautious_generate
        logger.info("[GPUCautiousDecoding] Registered `LLM.gpu_cautious_generate`.")
    except Exception as e:
        logger.warning(f"[GPUCautiousDecoding] Registration failed: {e}")
