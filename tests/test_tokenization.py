from __future__ import annotations

from collections.abc import Sequence

import pytest

from llm_pretrain.tokenization import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKEN_IDS,
    SYSTEM_TOKEN,
    UNK_TOKEN,
    USER_TOKEN,
    SentencePieceTokenizer,
    evaluate_tokenizer,
    write_source_balanced_tokenizer_corpus,
    write_tokenizer_corpus,
)


class FakeProcessor:
    def __init__(self, *, assistant_id: int = 6) -> None:
        self.pieces = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
            SYSTEM_TOKEN: 4,
            USER_TOKEN: 5,
            ASSISTANT_TOKEN: assistant_id,
        }

    def get_piece_size(self) -> int:
        return 24_576

    def pad_id(self) -> int:
        return 0

    def unk_id(self) -> int:
        return 1

    def bos_id(self) -> int:
        return 2

    def eos_id(self) -> int:
        return 3

    def piece_to_id(self, piece: str) -> int:
        return self.pieces.get(piece, 1)

    def encode(self, text: str, *, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [10 + ord(character) % 5 for character in text]

    def decode(self, ids: list[int]) -> str:
        return "decoded:" + ",".join(map(str, ids))

    def id_to_piece(self, token_id: int) -> str:
        return "piece" + str(token_id)


def test_special_token_contract_is_stable() -> None:
    assert SPECIAL_TOKEN_IDS == {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
    }
    tokenizer = SentencePieceTokenizer(FakeProcessor())
    assert tokenizer.encode("中", add_bos=True, add_eos=True)[::2] == [2, 3]


def test_incompatible_special_token_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="assistant"):
        SentencePieceTokenizer(FakeProcessor(assistant_id=9))


class MetricsTokenizer:
    eos_id = 3

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        del add_bos, add_eos
        return [100 if character == "坏" else 101 for character in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join("坏" if token_id == 100 else "好" for token_id in ids)

    def id_to_piece(self, token_id: int) -> str:
        return "<0xE5>" if token_id == 100 else "好"


def test_tokenizer_metrics_include_compression_and_byte_fallback() -> None:
    metrics = evaluate_tokenizer(MetricsTokenizer(), ["好坏", "好"])
    assert metrics.documents == 2
    assert metrics.total_tokens == 3
    assert metrics.total_characters == 3
    assert metrics.total_utf8_bytes == 9
    assert metrics.tokens_per_character == 1.0
    assert metrics.bytes_per_token == 3.0
    assert metrics.byte_fallback_rate == pytest.approx(1 / 3)


def test_empty_evaluation_is_well_defined() -> None:
    metrics = evaluate_tokenizer(MetricsTokenizer(), [])
    assert metrics.total_tokens == 0
    assert metrics.tokens_per_character == 0.0
    assert metrics.bytes_per_token == 0.0
    assert metrics.byte_fallback_rate == 0.0


def test_tokenizer_corpus_writer_uses_one_line_per_nonempty_document(tmp_path) -> None:
    path, count = write_tokenizer_corpus(["甲\n乙", "  ", "丙"], tmp_path / "train.txt")
    assert count == 2
    assert path.read_text(encoding="utf-8") == "甲 乙\n丙\n"


def test_source_balanced_tokenizer_corpus_is_text_only_and_bounded(tmp_path) -> None:
    documents = [
        ("zh", "中文甲"),
        ("zh", "中文乙"),
        ("wiki", "百科"),
        ("en", "hello"),
    ]

    path, stats = write_source_balanced_tokenizer_corpus(
        documents,
        tmp_path / "train.txt",
        source_weights={"zh": 0.8, "wiki": 0.1, "en": 0.1},
        max_utf8_bytes=100,
    )

    contents = path.read_text(encoding="utf-8")
    assert contents == "中文甲\n中文乙\n百科\nhello\n"
    assert stats.documents == 4
    assert stats.utf8_bytes == len(contents.encode("utf-8"))
    assert stats.source_documents == {"zh": 2, "wiki": 1, "en": 1}
    assert stats.utf8_bytes <= 100
