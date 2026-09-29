# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for GPUCautiousDecoder and GPUTreeTopology tree attention mask generation.
"""

import math
import torch
import pytest
from unittest.mock import MagicMock

from cautious_gpu.tree_attention import GPUTreeTopology
from cautious_gpu.gpu_decoder import GPUCautiousDecoder, _extract_topk_from_vllm_output


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 999

    def encode(self, text: str):
        return [101, 102]

    def decode(self, token_ids):
        return " ".join(str(tid) for tid in token_ids)


class MockCompletionOutput:
    def __init__(self, token_id, logprob_val):
        self.token_ids = [token_id]
        # Dict of token_id -> logprob
        self.logprobs = [{
            token_id: logprob_val,
            token_id + 1: logprob_val - 1.5,
            token_id + 2: logprob_val - 3.0,
        }]


class MockRequestOutput:
    def __init__(self, token_id, logprob_val):
        self.outputs = [MockCompletionOutput(token_id, logprob_val)]


class MockLLM:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.step_count = 0

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.step_count += len(prompts)
        outs = []
        for i, p in enumerate(prompts):
            pids = p["prompt_token_ids"]
            # Alternate confident (logprob -0.01 -> prob ~99%) and uncertain (logprob -1.2 -> prob ~30%)
            if len(pids) % 2 == 0:
                out = MockRequestOutput(token_id=200 + i, logprob_val=-0.01)
            else:
                out = MockRequestOutput(token_id=300 + i, logprob_val=-1.2)
            outs.append(out)
        return outs


def test_gpu_tree_topology_attention_mask():
    # Breadth 2, Depth 2:
    # 1 root + 2 depth1 + 4 depth2 = 7 nodes
    topo = GPUTreeTopology(breadth=2, depth=2, device="cpu")
    assert topo.num_total_nodes == 7
    assert topo.tree_parent_indices.tolist() == [-1, 0, 0, 1, 1, 2, 2]

    mask = topo.get_tree_attention_mask()
    assert mask.shape == (7, 7)

    # Self-attention
    for i in range(7):
        assert mask[i, i]

    # Root attention: all nodes attend to node 0
    for i in range(7):
        assert mask[i, 0]

    # Node 3 is child of 1, should attend to 1 and 0, but NOT 2, 4, 5, 6
    assert mask[3, 1] and mask[3, 0]
    assert not mask[3, 2]
    assert not mask[3, 4]
    assert not mask[3, 5]


def test_extract_topk_from_vllm_output():
    out = MockRequestOutput(token_id=42, logprob_val=-0.05)
    cand_ids, cand_lps, top1_prob = _extract_topk_from_vllm_output(out, breadth=3)
    assert len(cand_ids) == 3
    assert cand_ids[0] == 42
    assert top1_prob > 0.90


def test_gpu_cautious_decoder_adaptive():
    mock_llm = MockLLM()
    decoder = GPUCautiousDecoder(
        llm=mock_llm,
        breadth=2,
        depth=2,
        max_tokens=6,
        adaptive_cautious=True,
        confidence_threshold=0.85,
        commit_lookahead=True,
    )

    result = decoder.generate(prompt="Hello", verbose=False)
    assert result["num_committed_tokens"] == 6
    assert result["tokens_per_second"] > 0
    stats = result["stats"]
    assert stats["adaptive_cautious"] is True
    assert stats["num_forward_passes"] > 0
    assert stats["num_greedy_shortcuts"] >= 0


def test_gpu_cautious_decoder_strict_lookahead():
    mock_llm = MockLLM()
    decoder = GPUCautiousDecoder(
        llm=mock_llm,
        breadth=2,
        depth=2,
        max_tokens=4,
        adaptive_cautious=False,
        commit_lookahead=True,
    )

    result = decoder.generate(prompt="Hello", verbose=False)
    assert result["num_committed_tokens"] == 4
    stats = result["stats"]
    assert stats["adaptive_cautious"] is False
    assert stats["num_prunings"] >= 1
