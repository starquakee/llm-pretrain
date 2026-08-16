"""SentencePiece training and tokenizer evaluation utilities.

The module deliberately imports :mod:`sentencepiece` only inside the small
entry points that need it.  This keeps configuration, data inspection and the
unit tests usable before the optional native dependency is installed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

VOCAB_SIZE = 24_576

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SPECIAL_TOKEN_IDS: dict[str, int] = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    BOS_TOKEN: 2,
    EOS_TOKEN: 3,
    SYSTEM_TOKEN: 4,
    USER_TOKEN: 5,
    ASSISTANT_TOKEN: 6,
}
ROLE_TOKENS = (SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN)


class TokenizerProtocol(Protocol):
    """The minimal tokenizer surface consumed by the data pipeline."""

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to integer token ids."""
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        """Decode integer token ids."""
        raise NotImplementedError

    def id_to_piece(self, token_id: int) -> str:
        """Return the serialized SentencePiece token for an id."""
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        """Return the end-of-document token id."""
        raise NotImplementedError


def _sentencepiece_module() -> Any:
    try:
        import sentencepiece as sentencepiece_module
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "SentencePiece is required for tokenizer training/loading; run `uv sync`."
        ) from exc
    return sentencepiece_module


class SentencePieceTokenizer:
    """A small, validated wrapper around ``SentencePieceProcessor``."""

    def __init__(self, processor: Any, *, validate_special_tokens: bool = True) -> None:
        self._processor = processor
        if validate_special_tokens:
            self.validate_special_tokens()

    @classmethod
    def from_file(cls, model_file: str | Path) -> SentencePieceTokenizer:
        """Load a tokenizer model without importing SentencePiece at module import time."""
        spm = _sentencepiece_module()
        processor = spm.SentencePieceProcessor(model_file=str(Path(model_file)))
        return cls(processor)

    @property
    def vocab_size(self) -> int:
        return int(self._processor.get_piece_size())

    @property
    def pad_id(self) -> int:
        return int(self._processor.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self._processor.unk_id())

    @property
    def bos_id(self) -> int:
        return int(self._processor.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self._processor.eos_id())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._processor.decode([int(token_id) for token_id in ids]))

    def id_to_piece(self, token_id: int) -> str:
        return str(self._processor.id_to_piece(int(token_id)))

    def piece_to_id(self, piece: str) -> int:
        return int(self._processor.piece_to_id(piece))

    def validate_special_tokens(self) -> None:
        """Fail early when a model does not obey the repository's fixed id contract."""
        actual = {piece: self.piece_to_id(piece) for piece in SPECIAL_TOKEN_IDS}
        mismatches = {
            piece: (expected, actual[piece])
            for piece, expected in SPECIAL_TOKEN_IDS.items()
            if actual[piece] != expected
        }
        if mismatches:
            details = ", ".join(
                f"{piece}: expected {expected}, got {found}"
                for piece, (expected, found) in mismatches.items()
            )
            raise ValueError(f"Tokenizer special-token ids are incompatible ({details})")


def _tokenizer_corpus_line(text: str) -> bytes | None:
    if not isinstance(text, str):
        raise TypeError("Tokenizer corpus entries must be strings")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not canonical:
        return None
    return (canonical.replace("\n", " ") + "\n").encode("utf-8")


def write_tokenizer_corpus(
    texts: Iterable[str],
    path: str | Path,
    *,
    max_utf8_bytes: int | None = None,
) -> tuple[Path, int]:
    """Write training-partition documents as SentencePiece input, atomically."""
    if max_utf8_bytes is not None and max_utf8_bytes <= 0:
        raise ValueError("max_utf8_bytes must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    document_count = 0
    written_bytes = 0
    try:
        with temporary.open("wb") as handle:
            for text in texts:
                line = _tokenizer_corpus_line(text)
                if line is None:
                    continue
                if max_utf8_bytes is not None and written_bytes + len(line) > max_utf8_bytes:
                    continue
                handle.write(line)
                document_count += 1
                written_bytes += len(line)
                if max_utf8_bytes is not None and written_bytes == max_utf8_bytes:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, document_count


@dataclass(frozen=True)
class TokenizerCorpusStats:
    documents: int
    utf8_bytes: int
    source_documents: dict[str, int]
    source_utf8_bytes: dict[str, int]


def write_source_balanced_tokenizer_corpus(
    documents: Iterable[tuple[str, str]],
    path: str | Path,
    *,
    source_weights: Mapping[str, float],
    max_utf8_bytes: int,
) -> tuple[Path, TokenizerCorpusStats]:
    """Write a bounded, deterministic text-only sample with per-source byte quotas.

    Prepared JSONL is source-sequential. Separate on-disk spools keep every
    configured source represented without retaining the sample in RAM or
    teaching SentencePiece JSON keys, hashes, and document identifiers.
    """
    if max_utf8_bytes <= 0:
        raise ValueError("max_utf8_bytes must be positive")
    if not source_weights or any(weight <= 0 for weight in source_weights.values()):
        raise ValueError("source_weights must contain positive weights")
    total_weight = sum(source_weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("source_weights must sum to 1.0")

    source_names = list(source_weights)
    quotas: dict[str, int] = {}
    allocated = 0
    for source in source_names[:-1]:
        quota = int(max_utf8_bytes * source_weights[source])
        quotas[source] = quota
        allocated += quota
    quotas[source_names[-1]] = max_utf8_bytes - allocated
    if any(quota <= 0 for quota in quotas.values()):
        raise ValueError("max_utf8_bytes is too small to represent every source")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts = dict.fromkeys(source_names, 0)
    byte_counts = dict.fromkeys(source_names, 0)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as scratch:
            scratch_path = Path(scratch)
            spool_paths = {
                source: scratch_path / f"source-{index:04d}.txt"
                for index, source in enumerate(source_names)
            }
            with ExitStack() as stack:
                handles = {
                    source: stack.enter_context(spool_paths[source].open("wb"))
                    for source in source_names
                }
                for source, text in documents:
                    if source not in quotas:
                        raise ValueError(f"Tokenizer document has unknown source: {source}")
                    line = _tokenizer_corpus_line(text)
                    if line is None:
                        continue
                    if byte_counts[source] + len(line) > quotas[source]:
                        continue
                    handles[source].write(line)
                    counts[source] += 1
                    byte_counts[source] += len(line)
                    if all(
                        byte_counts[name] >= max(1, int(quotas[name] * 0.99))
                        for name in source_names
                    ):
                        break

            missing = [source for source, count in counts.items() if count == 0]
            if missing:
                raise ValueError(f"Tokenizer sample has no usable documents for sources: {missing}")
            with temporary.open("wb") as destination:
                for source in source_names:
                    with spool_paths[source].open("rb") as source_file:
                        shutil.copyfileobj(source_file, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return output, TokenizerCorpusStats(
        documents=sum(counts.values()),
        utf8_bytes=sum(byte_counts.values()),
        source_documents=counts,
        source_utf8_bytes=byte_counts,
    )


def train_sentencepiece(
    input_files: Sequence[str | Path],
    model_prefix: str | Path,
    *,
    vocab_size: int = VOCAB_SIZE,
    character_coverage: float = 0.99995,
    num_threads: int | None = None,
    hard_vocab_limit: bool = True,
) -> tuple[Path, Path]:
    """Train the project's BPE tokenizer and return model and vocabulary paths.

    ``input_files`` must point only at the training partition.  The function
    does not accept validation text by design, which makes tokenizer leakage a
    caller-visible mistake rather than hidden pipeline behavior.
    """
    if not input_files:
        raise ValueError("At least one tokenizer training file is required")
    if vocab_size < 263:  # 256 byte pieces plus the seven fixed special pieces
        raise ValueError("vocab_size is too small for byte fallback and fixed tokens")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError("character_coverage must be in (0, 1]")

    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    inputs = [str(Path(path)) for path in input_files]
    missing = [path for path in inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Tokenizer input files do not exist: {missing}")

    spm = _sentencepiece_module()
    kwargs: dict[str, Any] = {
        "input": inputs,
        "model_prefix": str(prefix),
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "byte_fallback": True,
        "normalization_rule_name": "nmt_nfkc_cf",
        "remove_extra_whitespaces": False,
        "split_digits": True,
        "allow_whitespace_only_pieces": True,
        "pad_id": SPECIAL_TOKEN_IDS[PAD_TOKEN],
        "unk_id": SPECIAL_TOKEN_IDS[UNK_TOKEN],
        "bos_id": SPECIAL_TOKEN_IDS[BOS_TOKEN],
        "eos_id": SPECIAL_TOKEN_IDS[EOS_TOKEN],
        "pad_piece": PAD_TOKEN,
        "unk_piece": UNK_TOKEN,
        "bos_piece": BOS_TOKEN,
        "eos_piece": EOS_TOKEN,
        "user_defined_symbols": list(ROLE_TOKENS),
        "hard_vocab_limit": hard_vocab_limit,
        "shuffle_input_sentence": False,
    }
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    spm.SentencePieceTrainer.Train(**kwargs)

    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    tokenizer = SentencePieceTokenizer.from_file(model_path)
    if hard_vocab_limit and tokenizer.vocab_size != vocab_size:
        raise RuntimeError(
            f"SentencePiece produced {tokenizer.vocab_size} pieces, expected {vocab_size}"
        )
    return model_path, vocab_path


@dataclass(frozen=True)
class TokenizerMetrics:
    documents: int
    total_tokens: int
    total_characters: int
    total_utf8_bytes: int
    tokens_per_character: float
    bytes_per_token: float
    byte_fallback_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _is_byte_piece(piece: str) -> bool:
    return len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")


def evaluate_tokenizer(tokenizer: TokenizerProtocol, texts: Iterable[str]) -> TokenizerMetrics:
    """Compute compression and byte-fallback metrics over a held-out iterable."""
    documents = 0
    total_tokens = 0
    total_characters = 0
    total_utf8_bytes = 0
    byte_tokens = 0

    for text in texts:
        documents += 1
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_characters += len(text)
        total_utf8_bytes += len(text.encode("utf-8"))
        byte_tokens += sum(_is_byte_piece(tokenizer.id_to_piece(token_id)) for token_id in ids)

    tokens_per_character = total_tokens / total_characters if total_characters else 0.0
    bytes_per_token = total_utf8_bytes / total_tokens if total_tokens else 0.0
    byte_fallback_rate = byte_tokens / total_tokens if total_tokens else 0.0
    return TokenizerMetrics(
        documents=documents,
        total_tokens=total_tokens,
        total_characters=total_characters,
        total_utf8_bytes=total_utf8_bytes,
        tokens_per_character=tokens_per_character,
        bytes_per_token=bytes_per_token,
        byte_fallback_rate=byte_fallback_rate,
    )
