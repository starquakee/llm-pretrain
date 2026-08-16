"""Validation-loss and fixed-prompt evaluation for language models."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .generation import GenerationConfig, GenerationResult, generate_text

FIXED_EVALUATION_PROMPTS: tuple[str, ...] = (
    "中国的首都是",
    "请用一句话解释什么是机器学习。",
    "春天来了，",  # noqa: RUF001 - intentional Chinese punctuation
    "用户：你好！\n助手：",  # noqa: RUF001 - intentional Chinese punctuation
)


@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: float
    perplexity: float
    tokens: int
    batches: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    validation: LossMetrics
    generations: tuple[GenerationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation": self.validation.to_dict(),
            "generations": [asdict(generation) for generation in self.generations],
        }


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


def _device_of(model: nn.Module, explicit: str | torch.device | None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _unpack_batch(batch: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Tensor):
        tokens = batch.to(device=device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[1] < 2:
            raise ValueError("a token-only evaluation batch needs at least two tokens")
        return tokens[:, :-1], tokens[:, 1:]
    if isinstance(batch, Mapping):
        if "input_ids" not in batch:
            raise KeyError("evaluation batch mapping is missing input_ids")
        inputs = torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels_value = batch.get("labels")
        if labels_value is None:
            if inputs.shape[-1] < 2:
                raise ValueError("a batch without labels needs at least two tokens")
            return inputs[:, :-1], inputs[:, 1:]
        labels = torch.as_tensor(labels_value, dtype=torch.long, device=device)
        if inputs.shape == labels.shape:
            if inputs.shape[-1] < 2:
                raise ValueError("language-model labels need at least two positions")
            return inputs[:, :-1], labels[:, 1:]
        return inputs, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        inputs = torch.as_tensor(batch[0], dtype=torch.long, device=device)
        labels = torch.as_tensor(batch[1], dtype=torch.long, device=device)
        if inputs.shape == labels.shape:
            if inputs.shape[-1] < 2:
                raise ValueError("language-model labels need at least two positions")
            return inputs[:, :-1], labels[:, 1:]
        return inputs, labels
    raise TypeError("batch must be a tensor, (input_ids, labels), or mapping")


def _autocast_context(device: torch.device, use_bf16: bool) -> Any:
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_loss(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    use_bf16: bool = True,
    ignore_index: int = -100,
) -> LossMetrics:
    """Compute token-weighted held-out cross entropy.

    A tensor-only batch is interpreted as packed next-token data and shifted by
    one token.  Explicit labels are used as-is, which is required for SFT's
    assistant-only ``-100`` mask.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    target_device = _device_of(model, device)
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if max_batches is not None and batch_count >= max_batches:
                    break
                input_ids, labels = _unpack_batch(batch, target_device)
                with _autocast_context(target_device, use_bf16):
                    logits = _extract_logits(model(input_ids))
                if logits.ndim != 3:
                    raise ValueError("model logits must have shape [batch, sequence, vocabulary]")
                if logits.shape[:2] != labels.shape:
                    raise ValueError(
                        "logit and label batch/sequence dimensions differ: "
                        f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
                    )
                valid_tokens = int(labels.ne(ignore_index).sum().item())
                if valid_tokens:
                    loss_sum = functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=ignore_index,
                        reduction="sum",
                    )
                    if not bool(torch.isfinite(loss_sum)):
                        raise FloatingPointError("non-finite validation loss")
                    total_loss += float(loss_sum.item())
                    total_tokens += valid_tokens
                batch_count += 1
    finally:
        model.train(was_training)

    if batch_count == 0:
        raise ValueError("evaluation received no batches")
    if total_tokens == 0:
        raise ValueError("evaluation labels contain no trainable tokens")
    mean_loss = total_loss / total_tokens
    # exp(>709) overflows a Python float; infinity is honest and JSON-friendly.
    perplexity = math.exp(mean_loss) if mean_loss < 709 else math.inf
    return LossMetrics(mean_loss, perplexity, total_tokens, batch_count)


def evaluate_prompts(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str] = FIXED_EVALUATION_PROMPTS,
    *,
    generation_config: GenerationConfig | None = None,
) -> tuple[GenerationResult, ...]:
    """Generate deterministic continuations for a stable qualitative suite."""

    config = generation_config or GenerationConfig(
        max_new_tokens=64, temperature=0.0, top_k=None, seed=42
    )
    return tuple(generate_text(model, tokenizer, prompt, config) for prompt in prompts)


def evaluate(
    model: nn.Module,
    tokenizer: Any,
    validation_batches: Iterable[Any],
    *,
    prompts: Sequence[str] = FIXED_EVALUATION_PROMPTS,
    generation_config: GenerationConfig | None = None,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    use_bf16: bool = True,
) -> EvaluationResult:
    """Run quantitative validation and the fixed qualitative prompt suite."""

    validation = evaluate_loss(
        model,
        validation_batches,
        device=device,
        max_batches=max_batches,
        use_bf16=use_bf16,
    )
    generations = evaluate_prompts(model, tokenizer, prompts, generation_config=generation_config)
    return EvaluationResult(validation, generations)
