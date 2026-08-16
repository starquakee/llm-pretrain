"""Deterministic autoregressive generation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50
    seed: int = 42
    eos_token_id: int | None = None
    context_length: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be positive when provided")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt: str
    completion: str
    text: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict) and isinstance(output.get("logits"), Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model output must be logits tensor or expose a logits tensor")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_context_length(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    for owner in (config, model):
        if owner is None:
            continue
        for name in ("max_seq_len", "context_length", "block_size", "seq_len"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _token_ids_from_dynamic(value: object) -> list[int]:
    """Validate a dynamically typed tokenizer return value."""

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
    elif callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
    else:
        raise TypeError("tokenizer must be callable or provide encode()")
    return _token_ids_from_dynamic(encoded)


def _decode(tokenizer: Any, token_ids: Iterable[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    if not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must provide decode()")
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(ids))


def resolve_eos_token_id(tokenizer: Any, configured: int | None = None) -> int | None:
    if configured is not None:
        return int(configured)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_id", None)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("tokenizer EOS id must be an integer")
        return value
    piece_to_id = getattr(tokenizer, "piece_to_id", None)
    if callable(piece_to_id):
        for piece in ("</s>", "<|eos|>"):
            candidate_value = piece_to_id(piece)
            if isinstance(candidate_value, bool) or not isinstance(candidate_value, int):
                raise TypeError("tokenizer piece_to_id() must return an integer")
            candidate = candidate_value
            # SentencePiece returns unk_id for an unknown piece, so prefer only
            # the repository's fixed EOS id when no eos_id property exists.
            if candidate == 3:
                return candidate
    return None


def generate_tokens(
    model: nn.Module,
    input_ids: Tensor,
    config: GenerationConfig | None = None,
    **overrides: Any,
) -> Tensor:
    """Generate continuations for a batch, deterministically for a fixed seed.

    Greedy decoding is selected by ``temperature=0``.  Sampling uses a local
    generator and therefore does not change PyTorch's process-global RNG state.
    All batch rows stop together once every row has produced EOS; this keeps the
    tensor rectangular without silently inserting padding tokens.
    """

    settings = config or GenerationConfig()
    if overrides:
        values = {name: getattr(settings, name) for name in GenerationConfig.__dataclass_fields__}
        values.update(overrides)
        settings = GenerationConfig(**values)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, sequence] with a non-empty sequence")

    device = _model_device(model)
    generated = input_ids.to(device=device, dtype=torch.long)
    context_length = settings.context_length or _infer_context_length(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(settings.max_new_tokens):
                model_input = generated[:, -context_length:] if context_length else generated
                logits = _extract_logits(model(model_input))
                if logits.ndim != 3 or logits.shape[:2] != model_input.shape:
                    raise ValueError("model logits must have shape [batch, sequence, vocabulary]")
                next_logits = logits[:, -1, :].float()
                if settings.temperature == 0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / settings.temperature
                    if settings.top_k is not None:
                        k = min(settings.top_k, next_logits.shape[-1])
                        threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                        next_logits = next_logits.masked_fill(
                            next_logits < threshold, float("-inf")
                        )
                    probabilities = torch.softmax(next_logits, dim=-1)
                    if not bool(torch.isfinite(probabilities).all()):
                        raise FloatingPointError("non-finite sampling probabilities")
                    next_token = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    )

                if settings.eos_token_id is not None:
                    eos_fill = torch.full_like(next_token, settings.eos_token_id)
                    next_token = torch.where(finished[:, None], eos_fill, next_token)
                generated = torch.cat((generated, next_token), dim=1)
                if settings.eos_token_id is not None:
                    finished |= next_token.squeeze(1).eq(settings.eos_token_id)
                    if bool(finished.all()):
                        break
    finally:
        model.train(was_training)
    return generated


def generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    config: GenerationConfig | None = None,
    **overrides: Any,
) -> GenerationResult:
    """Tokenize a prompt, generate, and return both completion and full text."""

    settings = config or GenerationConfig()
    values = {name: getattr(settings, name) for name in GenerationConfig.__dataclass_fields__}
    values.update(overrides)
    values["eos_token_id"] = resolve_eos_token_id(tokenizer, values.get("eos_token_id"))
    settings = GenerationConfig(**values)
    prompt_ids = _encode(tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("the encoded prompt is empty")
    tokens = (
        generate_tokens(model, torch.tensor([prompt_ids], dtype=torch.long), settings)[0]
        .detach()
        .cpu()
        .tolist()
    )
    generated_ids = tokens[len(prompt_ids) :]
    stopped_on_eos = bool(
        settings.eos_token_id is not None
        and generated_ids
        and generated_ids[-1] == settings.eos_token_id
    )
    completion = _decode(tokenizer, generated_ids)
    return GenerationResult(
        prompt=prompt,
        completion=completion,
        text=_decode(tokenizer, tokens),
        token_ids=tuple(tokens),
        generated_token_ids=tuple(generated_ids),
        stopped_on_eos=stopped_on_eos,
    )


# A short ergonomic alias for library callers and the CLI implementation.
generate = generate_text


def interactive_generate(
    model: nn.Module,
    tokenizer: Any,
    config: GenerationConfig | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    prompt_prefix: str = "用户：",  # noqa: RUF001 - intentional Chinese punctuation
    response_prefix: str = "助手：",  # noqa: RUF001 - intentional Chinese punctuation
    exit_commands: tuple[str, ...] = ("exit", "quit", "/exit", "/quit", "退出"),
) -> list[GenerationResult]:
    """Run a testable interactive prompt loop and return its transcript."""

    transcript: list[GenerationResult] = []
    while True:
        try:
            user_text = input_fn(prompt_prefix)
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.strip().lower() in {command.lower() for command in exit_commands}:
            break
        if not user_text.strip():
            continue
        result = generate_text(model, tokenizer, user_text, config)
        transcript.append(result)
        output_fn(f"{response_prefix}{result.completion}")
    return transcript
