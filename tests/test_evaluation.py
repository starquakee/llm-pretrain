from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from llm_pretrain.evaluation import evaluate_loss, evaluate_prompts
from llm_pretrain.generation import GenerationConfig


class PerfectNextModel(nn.Module):
    def __init__(self, vocab_size: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*input_ids.shape, self.vocab_size), -10.0)
        targets = (input_ids + 1) % self.vocab_size
        return logits.scatter(-1, targets.unsqueeze(-1), 10.0) + self.weight * 0


class NumericTokenizer:
    eos_token_id = 5

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(value) for value in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(value) for value in ids)


def test_token_weighted_validation_loss_for_packed_batches() -> None:
    model = PerfectNextModel()
    metrics = evaluate_loss(
        model,
        [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 4]])],
        use_bf16=False,
    )
    assert metrics.tokens == 4
    assert metrics.batches == 2
    assert metrics.loss < 1e-6
    assert math.isclose(metrics.perplexity, 1.0, abs_tol=1e-5)


def test_mapping_labels_are_shifted_and_ignore_masked_prefix() -> None:
    model = PerfectNextModel()
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "labels": torch.tensor([[-100, -100, 2, 3]]),
    }
    metrics = evaluate_loss(model, [batch], use_bf16=False)
    assert metrics.tokens == 2
    assert metrics.loss < 1e-6


def test_empty_validation_is_rejected() -> None:
    with pytest.raises(ValueError, match="no batches"):
        evaluate_loss(PerfectNextModel(), [], use_bf16=False)


def test_fixed_prompt_generation_uses_supplied_suite() -> None:
    generations = evaluate_prompts(
        PerfectNextModel(),
        NumericTokenizer(),
        prompts=("0", "1"),
        generation_config=GenerationConfig(
            max_new_tokens=1, temperature=0, top_k=None, eos_token_id=None
        ),
    )
    assert [result.completion for result in generations] == ["1", "2"]
