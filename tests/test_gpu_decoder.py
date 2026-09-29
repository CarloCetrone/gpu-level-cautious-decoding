# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for pure GPUCautiousDecoder and GPUTreeTopology tree attention mask generation.
"""

import math
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import pytest

from cautious_gpu.tree_attention import GPUTreeTopology
from cautious_gpu.gpu_decoder import GPUCautiousDecoder


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
            out = MockRequestOutput(token_id=200 + i, logprob_val=-0.5)
            outs.append(out)
        return outs


class MockModelOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class MockPyTorchModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        new_past_kv = (torch.zeros(1),)
        return MockModelOutput(logits=logits, past_key_values=new_past_kv)


def test_gpu_tree_topology():
    # Breadth 2, Depth 2:
    # Level 1: 2 nodes (parent -1)
    # Level 2: 4 nodes (parent 0, 0, 1, 1)
    # Total candidate nodes = 2 + 4 = 6
    topo = GPUTreeTopology(breadth=2, depth=2, device="cpu")
    assert topo.num_candidate_nodes == 6
    assert topo.candidate_parent_indices.tolist() == [-1, -1, 0, 0, 1, 1]

    mask = topo.get_tree_attention_mask()
    assert mask.shape == (6, 6)

    # Self-attention must be True
    for i in range(6):
        assert mask[i, i]

    # Node 2 is child of 0
    assert mask[2, 0]
    # Node 2 must NOT attend to sibling 1 or cousin 4
    assert not mask[2, 1]
    assert not mask[2, 4]

    # Test 4D mask for prefix length 10
    mask_4d = topo.get_4d_tree_attention_mask(prefix_len=10)
    assert mask_4d.shape == (1, 1, 6, 16)
    # All candidate nodes attend to prefix
    assert torch.all(mask_4d[:, :, :, :10] == 0.0)

    # Test position IDs
    pos_ids = topo.get_tree_position_ids(prefix_len=10)
    assert pos_ids.shape == (1, 6)
    assert pos_ids[0, 0].item() == 10  # Depth 1
    assert pos_ids[0, 2].item() == 11  # Depth 2


def test_gpu_cautious_decoder_batched():
    mock_llm = MockLLM()
    decoder = GPUCautiousDecoder(
        llm=mock_llm,
        breadth=2,
        depth=2,
        max_tokens=4,
    )

    result = decoder.generate(prompt="Hello", verbose=False)
    assert result["num_committed_tokens"] == 4
    assert result["tokens_per_second"] > 0
    stats = result["stats"]
    assert stats["breadth"] == 2
    assert stats["depth"] == 2
    assert stats["num_prunings"] >= 1


def test_gpu_cautious_decoder_native_model():
    model = MockPyTorchModel()
    mock_llm = MagicMock()
    mock_llm.model = model
    mock_llm.get_tokenizer.return_value = MockTokenizer()

    decoder = GPUCautiousDecoder(
        llm=mock_llm,
        breadth=2,
        depth=2,
        max_tokens=3,
    )

    result = decoder.generate(prompt="Hello", verbose=False)
    assert result["num_committed_tokens"] == 3
    assert result["stats"]["execution_mode"] == "native_tree_attention"
    assert result["stats"]["num_prunings"] >= 1
