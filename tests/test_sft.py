from __future__ import annotations

import random

import pytest
import torch
from torch import nn

from llm_pretrain.sft import (
    SFTConfig,
    SFTDataset,
    SFTExample,
    assistant_only_loss,
    create_sft_optimizer,
    deterministic_coig_split,
    make_sft_collator,
    parse_coig_record,
    tokenize_sft_example,
    train_sft,
)


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 29) for character in text]


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, width: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def _records(count: int) -> list[dict[str, str]]:
    return [
        {"instruction": f"问题 {index}", "input": "背景", "output": f"回答 {index}"}
        for index in range(count)
    ]


def test_coig_selection_is_deterministic_and_order_independent() -> None:
    records = _records(12)
    shuffled = records.copy()
    random.Random(99).shuffle(shuffled)
    first = deterministic_coig_split(records, train_size=8, validation_size=2, seed=7)
    second = deterministic_coig_split(shuffled, train_size=8, validation_size=2, seed=7)
    assert [item.identifier for item in first.train] == [item.identifier for item in second.train]
    assert set(first.train).isdisjoint(first.validation)


def test_coig_schema_parsing_and_invalid_records() -> None:
    parsed = parse_coig_record(
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！"},  # noqa: RUF001
            ]
        }
    )
    assert parsed is not None and parsed.user == "你好" and parsed.assistant == "你好!"
    assert parse_coig_record({"instruction": "missing answer"}) is None


def test_tokenization_masks_every_non_assistant_token() -> None:
    tokenizer = CharacterTokenizer()
    example = SFTExample("系统", "问题", "答案", "id")
    encoded = tokenize_sft_example(example, tokenizer, max_length=128)
    response_size = len(tokenizer.encode("答案")) + 1
    assert encoded["labels"][:-response_size] == [-100] * (len(encoded["labels"]) - response_size)
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    assert encoded["input_ids"] == [
        token if label == -100 else label
        for token, label in zip(encoded["input_ids"], encoded["labels"], strict=True)
    ]


def test_collator_uses_minus_100_for_label_padding() -> None:
    collate = make_sft_collator(0)
    batch = collate(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [1], "labels": [1]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [1, -100]]
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]


def test_sft_defaults_use_adamw_and_two_epochs() -> None:
    config = SFTConfig()
    optimizer = create_sft_optimizer(TinyLM(), config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert config.epochs == 2


def test_assistant_loss_and_cpu_training_smoke() -> None:
    torch.manual_seed(0)
    model = TinyLM()
    dataset = SFTDataset(
        [SFTExample("s", "u", "a", "1"), SFTExample("s", "v", "b", "2")],
        CharacterTokenizer(),
        max_length=32,
    )
    collate = make_sft_collator(0)
    batch = collate([dataset[0], dataset[1]])
    initial = float(assistant_only_loss(model(batch["input_ids"]), batch["labels"]).item())
    result = train_sft(
        model,
        [batch],
        validation_batches=[batch],
        config=SFTConfig(learning_rate=0.05, epochs=2, max_length=32),
        device="cpu",
    )
    final = float(assistant_only_loss(model(batch["input_ids"]), batch["labels"]).item())
    assert result.optimizer_steps == 2
    assert len(result.history) == 2
    assert final < initial
