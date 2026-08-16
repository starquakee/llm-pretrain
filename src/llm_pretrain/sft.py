"""Deterministic COIG preparation and full-parameter supervised fine-tuning."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SFTConfig:
    learning_rate: float = 5e-5
    epochs: int = 2
    max_length: int = 1024
    train_examples: int = 50_000
    validation_examples: int = 1_000
    seed: int = 42
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0 or self.max_length < 2:
            raise ValueError("epochs must be positive and max_length must be at least two")
        if self.train_examples < 0 or self.validation_examples < 0:
            raise ValueError("split sizes must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True, slots=True)
class SFTExample:
    system: str
    user: str
    assistant: str
    identifier: str


@dataclass(frozen=True, slots=True)
class SFTSplit:
    train: tuple[SFTExample, ...]
    validation: tuple[SFTExample, ...]


@dataclass(frozen=True, slots=True)
class SFTEpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float | None
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class SFTTrainResult:
    history: tuple[SFTEpochMetrics, ...]
    optimizer_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "history": [asdict(item) for item in self.history],
        }


DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"
_WHITESPACE = re.compile(r"\s+")


def normalize_sft_text(value: Any) -> str:
    """Apply the same stable Unicode/whitespace normalization before hashing."""

    if value is None:
        return ""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(value))).strip()


def _from_messages(record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    raw_messages = record.get("messages") or record.get("conversations")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        return None
    system = ""
    user = ""
    assistant = ""
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            continue
        role = normalize_sft_text(raw.get("role") or raw.get("from")).lower()
        content = normalize_sft_text(raw.get("content") or raw.get("value"))
        if role in {"system"} and content and not system:
            system = content
        elif role in {"user", "human"} and content and not user:
            user = content
        elif role in {"assistant", "gpt", "bot"} and content and user:
            assistant = content
            break
    if user and assistant:
        return system, user, assistant
    return None


def parse_coig_record(record: Mapping[str, Any]) -> SFTExample | None:
    """Convert common COIG schemas into one single-turn chat example."""

    messages = _from_messages(record)
    if messages is not None:
        system, user, assistant = messages
    else:
        instruction = normalize_sft_text(
            record.get("instruction") or record.get("prompt") or record.get("question")
        )
        extra_input = normalize_sft_text(record.get("input") or record.get("context"))
        user = f"{instruction}\n{extra_input}".strip() if extra_input else instruction
        assistant = normalize_sft_text(
            record.get("output") or record.get("response") or record.get("answer")
        )
        system = normalize_sft_text(record.get("system"))
    if not user or not assistant:
        return None
    system = system or DEFAULT_SYSTEM_PROMPT
    digest_source = "\0".join((system, user, assistant)).encode("utf-8")
    identifier = hashlib.sha256(digest_source).hexdigest()
    return SFTExample(system, user, assistant, identifier)


def deterministic_coig_split(
    records: Iterable[Mapping[str, Any]],
    *,
    train_size: int = 50_000,
    validation_size: int = 1_000,
    seed: int = 42,
) -> SFTSplit:
    """Deduplicate and select COIG samples independent of input iteration order."""

    if train_size < 0 or validation_size < 0:
        raise ValueError("split sizes must be non-negative")
    unique: dict[str, SFTExample] = {}
    for record in records:
        example = parse_coig_record(record)
        if example is not None:
            unique.setdefault(example.identifier, example)
    required = train_size + validation_size
    if len(unique) < required:
        raise ValueError(f"COIG has only {len(unique)} usable unique examples; {required} required")
    seed_bytes = str(seed).encode("ascii")
    ranked = sorted(
        unique.values(),
        key=lambda item: hashlib.sha256(seed_bytes + bytes.fromhex(item.identifier)).digest(),
    )[:required]
    return SFTSplit(
        train=tuple(ranked[:train_size]),
        validation=tuple(ranked[train_size:]),
    )


# Name used by data-oriented callers.
prepare_coig_splits = deterministic_coig_split


def format_chat_prefix(example: SFTExample) -> str:
    """Return the prompt portion, ending immediately after the assistant marker."""

    return f"<|system|>\n{example.system}\n<|user|>\n{example.user}\n<|assistant|>\n"


def format_chat_example(example: SFTExample) -> str:
    return f"{format_chat_prefix(example)}{example.assistant}</s>"


def _token_ids_from_dynamic(value: object) -> list[int]:
    """Validate token ids returned through a dynamic tokenizer API."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("tokenizer mapping is missing input_ids")
        value = value["input_ids"]
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("tokenizer input_ids must be a sequence of integers")
    values: Sequence[object] = value
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        values = values[0]
    token_ids: list[int] = []
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("tokenizer input_ids must contain only integers")
        token_ids.append(token_id)
    return token_ids


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded: object
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
    elif hasattr(tokenizer, "EncodeAsIds"):
        encoded = tokenizer.EncodeAsIds(text)
    elif callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
    else:
        raise TypeError("tokenizer must provide encode(), EncodeAsIds(), or be callable")
    return _token_ids_from_dynamic(encoded)


def _special_token_id(tokenizer: Any, attribute: str, piece: str) -> int | None:
    value = getattr(tokenizer, attribute, None)
    if value is None and attribute == "eos_token_id":
        value = getattr(tokenizer, "eos_id", None)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"tokenizer {attribute} must be an integer")
        return value
    piece_to_id = getattr(tokenizer, "piece_to_id", None)
    if callable(piece_to_id):
        piece_value = piece_to_id(piece)
        if isinstance(piece_value, bool) or not isinstance(piece_value, int):
            raise TypeError("tokenizer piece_to_id() must return an integer")
        return piece_value if piece_value >= 0 else None
    return None


def tokenize_sft_example(
    example: SFTExample,
    tokenizer: Any,
    *,
    max_length: int = 1024,
) -> dict[str, list[int]]:
    """Tokenize one chat with loss enabled only for the assistant response."""

    if max_length < 2:
        raise ValueError("max_length must be at least two")
    prefix_ids = _encode(tokenizer, format_chat_prefix(example))
    response_ids = _encode(tokenizer, example.assistant)
    eos_id = _special_token_id(tokenizer, "eos_token_id", "</s>")
    if eos_id is not None:
        response_ids.append(eos_id)
    if not response_ids:
        raise ValueError("assistant response produced no tokens")

    # Preserve the beginning of the response and EOS when it alone exceeds the
    # window. Otherwise trim the oldest prompt tokens; the assistant marker is
    # at the prompt tail and therefore remains present.
    if len(response_ids) > max_length:
        response_ids = response_ids[:max_length]
        if eos_id is not None:
            response_ids[-1] = eos_id
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[-(max_length - len(response_ids)) :]
    input_ids = prefix_ids + response_ids
    labels = [-100] * len(prefix_ids) + response_ids.copy()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class SFTDataset(Dataset[dict[str, list[int]]]):
    def __init__(
        self,
        examples: Sequence[SFTExample],
        tokenizer: Any,
        *,
        max_length: int = 1024,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return tokenize_sft_example(
            self.examples[index], self.tokenizer, max_length=self.max_length
        )


def make_sft_collator(
    pad_token_id: int,
) -> Any:
    """Create a right-padding collator with ``-100`` label padding."""

    def collate(items: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Tensor]:
        if not items:
            raise ValueError("cannot collate an empty SFT batch")
        width = max(len(item["input_ids"]) for item in items)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for item in items:
            size = len(item["input_ids"])
            padding = width - size
            input_rows.append([int(v) for v in item["input_ids"]] + [pad_token_id] * padding)
            label_rows.append([int(v) for v in item["labels"]] + [-100] * padding)
            mask_rows.append([1] * size + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.bool),
        }

    return collate


def create_sft_optimizer(model: nn.Module, config: SFTConfig | None = None) -> torch.optim.AdamW:
    """Create the specified full-parameter AdamW optimizer (never Muon)."""

    settings = config or SFTConfig()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=settings.learning_rate,
        betas=settings.betas,
        weight_decay=settings.weight_decay,
    )


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model output must be logits tensor or expose a logits tensor")


def assistant_only_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels must have shapes [B, T, V] and [B, T]")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if not bool(shifted_labels.ne(-100).any()):
        raise ValueError("SFT batch has no assistant response labels")
    return functional.cross_entropy(
        shifted_logits.float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )


def _forward(
    model: nn.Module, batch: Mapping[str, Any], device: torch.device
) -> tuple[Tensor, Tensor]:
    input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
    attention = batch.get("attention_mask")
    try:
        output = model(
            input_ids,
            attention_mask=(
                torch.as_tensor(attention, dtype=torch.bool, device=device)
                if attention is not None
                else None
            ),
        )
    except TypeError:
        output = model(input_ids)
    return _extract_logits(output), labels


def _validation_loss(
    model: nn.Module, batches: Iterable[Mapping[str, Any]], device: torch.device
) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits, labels = _forward(model, batch, device)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            tokens = int(shifted_labels.ne(-100).sum().item())
            if tokens:
                total_loss += float(
                    functional.cross_entropy(
                        shifted_logits.float().reshape(-1, logits.shape[-1]),
                        shifted_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ).item()
                )
                total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("SFT validation contains no assistant tokens")
    return total_loss / total_tokens


def train_sft(
    model: nn.Module,
    train_batches: Iterable[Mapping[str, Any]],
    *,
    validation_batches: Iterable[Mapping[str, Any]] | None = None,
    config: SFTConfig | None = None,
    device: str | torch.device | None = None,
) -> SFTTrainResult:
    """Run two-epoch full-parameter SFT with assistant-only labels by default."""

    settings = config or SFTConfig()
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(target_device)
    optimizer = create_sft_optimizer(model, settings)
    history: list[SFTEpochMetrics] = []
    optimizer_steps = 0

    for epoch in range(1, settings.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        batch_count = 0
        pending = 0
        for batch in train_batches:
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if target_device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits, labels = _forward(model, batch, target_device)
                loss = assistant_only_loss(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite SFT loss")
            (loss / settings.gradient_accumulation_steps).backward()
            epoch_loss += float(loss.detach().item())
            batch_count += 1
            pending += 1
            if pending == settings.gradient_accumulation_steps:
                if settings.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
        if batch_count == 0:
            raise ValueError("SFT training received no batches")
        if pending:
            # Undo the division by the requested accumulation count for a short
            # final group so its gradient remains an average of that group.
            scale = settings.gradient_accumulation_steps / pending
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            if settings.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        validation_loss = (
            _validation_loss(model, validation_batches, target_device)
            if validation_batches is not None
            else None
        )
        history.append(
            SFTEpochMetrics(epoch, epoch_loss / batch_count, validation_loss, optimizer_steps)
        )
    return SFTTrainResult(tuple(history), optimizer_steps)
