from __future__ import annotations

import torch
from torch import nn

from llm_pretrain.generation import (
    GenerationConfig,
    generate_text,
    generate_tokens,
    interactive_generate,
)


class NextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 8, forced_token: int | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.forced_token = forced_token

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*input_ids.shape, self.vocab_size), -20.0, device=input_ids.device)
        targets = (
            torch.full_like(input_ids, self.forced_token)
            if self.forced_token is not None
            else (input_ids + 1) % self.vocab_size
        )
        return logits.scatter(-1, targets.unsqueeze(-1), 20.0) + self.anchor * 0


class FakeTokenizer:
    eos_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(piece) for piece in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [token for token in ids if token != self.eos_token_id]
        return " ".join(str(token) for token in ids)


def test_greedy_generation_and_mode_restoration() -> None:
    model = NextTokenModel()
    model.train()
    result = generate_tokens(
        model,
        torch.tensor([[1, 2]]),
        GenerationConfig(max_new_tokens=3, temperature=0, top_k=None),
    )
    assert result.tolist() == [[1, 2, 3, 4, 5]]
    assert model.training


def test_eos_stops_text_generation() -> None:
    model = NextTokenModel(forced_token=7)
    result = generate_text(
        model,
        FakeTokenizer(),
        "1 2",
        GenerationConfig(max_new_tokens=10, temperature=0, eos_token_id=7),
    )
    assert result.generated_token_ids == (7,)
    assert result.stopped_on_eos


def test_seeded_sampling_is_repeatable() -> None:
    class UniformModel(NextTokenModel):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((*input_ids.shape, self.vocab_size)) + self.anchor * 0

    model = UniformModel()
    config = GenerationConfig(max_new_tokens=8, temperature=1, top_k=5, seed=123)
    first = generate_tokens(model, torch.tensor([[1]]), config)
    second = generate_tokens(model, torch.tensor([[1]]), config)
    assert torch.equal(first, second)


def test_interactive_helper_is_injectable() -> None:
    responses = iter(["1", "退出"])
    output: list[str] = []
    transcript = interactive_generate(
        NextTokenModel(forced_token=7),
        FakeTokenizer(),
        GenerationConfig(max_new_tokens=2, temperature=0, eos_token_id=7),
        input_fn=lambda _prompt: next(responses),
        output_fn=output.append,
    )
    assert len(transcript) == 1
    assert output == ["助手："]  # noqa: RUF001 - intentional Chinese punctuation
