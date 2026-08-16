"""Strict, serialisable configuration objects for the training pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, TypeVar, Union, cast, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


ConfigT = TypeVar("ConfigT", bound="YamlConfig")


class YamlConfig:
    """Mixin providing strict YAML loading and validation for dataclasses."""

    _yaml_module: ClassVar[Any | None] = None

    def validate(self) -> None:
        """Validate this configuration.

        Subclasses override this method.  It is intentionally public so callers
        can validate configurations that they have modified programmatically.
        """

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe representation of the configuration."""

        def normalise(value: Any) -> Any:
            if isinstance(value, tuple):
                return [normalise(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: normalise(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalise(item) for item in value]
            return value

        return normalise(asdict(cast(Any, self)))

    @classmethod
    def from_dict(cls: type[ConfigT], values: Mapping[str, Any]) -> ConfigT:
        """Build ``cls`` from a mapping, rejecting unknown keys and bad types."""

        if not isinstance(values, Mapping):
            raise ConfigError(f"{cls.__name__} must be loaded from a mapping")

        field_names = {field.name for field in fields(cast(Any, cls))}
        unknown = sorted(set(values) - field_names)
        if unknown:
            raise ConfigError(f"unknown {cls.__name__} field(s): {', '.join(unknown)}")

        type_hints = get_type_hints(cls)
        converted: dict[str, Any] = {}
        for name, value in values.items():
            converted[name] = _coerce_value(value, type_hints[name], name)

        try:
            result = cls(**converted)
        except TypeError as exc:
            raise ConfigError(f"invalid {cls.__name__}: {exc}") from exc
        result.validate()
        return result

    @classmethod
    def load_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        """Load and strictly validate a YAML mapping."""

        yaml = cls._get_yaml()
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as handle:
                values = yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigError(f"cannot read configuration {source}: {exc}") from exc
        if values is None:
            values = {}
        return cls.from_dict(values)

    def save_yaml(self, path: str | Path) -> None:
        """Validate and atomically save this configuration as YAML."""

        self.validate()
        yaml = self._get_yaml()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(
                    self.to_dict(),
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ConfigError(f"cannot write configuration {destination}: {exc}") from exc

    @classmethod
    def _get_yaml(cls) -> Any:
        if cls._yaml_module is None:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - packaging failure
                raise RuntimeError("PyYAML is required for YAML configuration support") from exc
            cls._yaml_module = yaml
        return cls._yaml_module


def _coerce_value(value: Any, expected: Any, name: str) -> Any:
    """Perform the small set of safe conversions needed after YAML parsing."""

    origin = get_origin(expected)
    args = get_args(expected)

    if origin in (Union, UnionType):
        for option in args:
            try:
                return _coerce_value(value, option, name)
            except ConfigError:
                pass
        raise ConfigError(f"field {name!r} does not match {expected!r}")

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"field {name!r} must be a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(item, args[0], name) for item in value)
        if args and len(value) != len(args):
            raise ConfigError(f"field {name!r} must contain {len(args)} items")
        return tuple(
            _coerce_value(item, item_type, name)
            for item, item_type in zip(value, args, strict=True)
        )

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"field {name!r} must be a mapping")
        key_type, value_type = args
        return {
            _coerce_value(key, key_type, name): _coerce_value(item, value_type, name)
            for key, item in value.items()
        }

    if expected is Any:
        return value
    if expected is bool:
        if type(value) is not bool:
            raise ConfigError(f"field {name!r} must be a boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ConfigError(f"field {name!r} must be an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"field {name!r} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"field {name!r} must be a string")
        return value
    if expected is type(None):
        if value is not None:
            raise ConfigError(f"field {name!r} must be null")
        return None
    if isinstance(expected, type) and not isinstance(value, expected):
        raise ConfigError(f"field {name!r} must be {expected.__name__}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(slots=True)
class ModelConfig(YamlConfig):
    """Decoder-only transformer architecture."""

    vocab_size: int = 24_576
    max_seq_len: int = 1_024
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    intermediate_size: int = 1_920
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        _require(self.vocab_size >= 8, "vocab_size must be at least 8")
        _require(self.max_seq_len >= 2, "max_seq_len must be at least 2")
        _require(self.n_layers >= 1, "n_layers must be positive")
        _require(self.d_model >= 2, "d_model must be at least 2")
        _require(self.n_heads >= 1, "n_heads must be positive")
        _require(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads",
        )
        _require(self.head_dim % 2 == 0, "attention head_dim must be even for RoPE")
        _require(self.intermediate_size >= 1, "intermediate_size must be positive")
        _require(self.rope_theta > 0.0, "rope_theta must be positive")
        _require(self.rms_norm_eps > 0.0, "rms_norm_eps must be positive")
        _require(self.dropout == 0.0, "this architecture requires dropout=0")
        _require(not self.bias, "this architecture requires bias=false")


@dataclass(slots=True)
class DataConfig(YamlConfig):
    """Dataset, tokenizer, and packed-shard settings."""

    data_dir: str = "../llm-pretrain-data"
    sequence_length: int = 1_024
    tokenizer_vocab_size: int = 24_576
    validation_tokens: int = 10_000_000
    shard_size_tokens: int = 100_000_000
    chinese_web_ratio: float = 0.8
    chinese_wikipedia_ratio: float = 0.1
    english_web_ratio: float = 0.1
    seed: int = 1_337
    num_workers: int = 4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require(bool(self.data_dir.strip()), "data_dir must not be empty")
        _require(self.sequence_length >= 2, "sequence_length must be at least 2")
        _require(self.tokenizer_vocab_size >= 256, "tokenizer_vocab_size is too small")
        _require(self.validation_tokens >= 1, "validation_tokens must be positive")
        _require(self.shard_size_tokens >= self.sequence_length, "shard is too small")
        ratios = (
            self.chinese_web_ratio,
            self.chinese_wikipedia_ratio,
            self.english_web_ratio,
        )
        _require(all(ratio >= 0.0 for ratio in ratios), "data ratios cannot be negative")
        _require(abs(sum(ratios) - 1.0) < 1e-9, "data ratios must sum to 1")
        _require(self.seed >= 0, "seed cannot be negative")
        _require(self.num_workers >= 0, "num_workers cannot be negative")


@dataclass(slots=True)
class OptimizerConfig(YamlConfig):
    """Muon experiment and AdamW baseline hyperparameters."""

    optimizer: str = "muon"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    muon_lr_adjustment: str = "original"
    auxiliary_lr: float = 3e-4
    auxiliary_betas: tuple[float, float] = (0.9, 0.95)
    auxiliary_weight_decay: float = 0.01
    adamw_lr: float = 6e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require(self.optimizer in {"muon", "adamw"}, "optimizer must be muon or adamw")
        for name in (
            "muon_lr",
            "auxiliary_lr",
            "adamw_lr",
        ):
            _require(getattr(self, name) > 0.0, f"{name} must be positive")
        _require(0.0 <= self.muon_momentum < 1.0, "invalid Muon momentum")
        _require(self.muon_ns_steps >= 1, "muon_ns_steps must be positive")
        _require(
            self.muon_lr_adjustment in {"original", "spectral_norm", "match_rms_adamw"},
            "unsupported muon_lr_adjustment",
        )
        for name in ("muon_weight_decay", "auxiliary_weight_decay", "adamw_weight_decay"):
            _require(getattr(self, name) >= 0.0, f"{name} cannot be negative")
        for name in ("auxiliary_betas", "adamw_betas"):
            betas = getattr(self, name)
            _require(
                len(betas) == 2 and all(0.0 <= beta < 1.0 for beta in betas),
                f"{name} must contain two values in [0, 1)",
            )
        _require(0.0 <= self.warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)")
        _require(0.0 <= self.min_lr_ratio <= 1.0, "min_lr_ratio must be in [0, 1]")


@dataclass(slots=True)
class TrainConfig(YamlConfig):
    """Runtime settings shared by pretraining and A/B experiments."""

    output_dir: str = "../llm-pretrain-runs"
    seed: int = 1_337
    max_tokens: int = 2_000_000_000
    time_budget_hours: float = 120.0
    global_batch_tokens: int = 131_072
    micro_batch_sequences: int = 0
    micro_batch_candidates: tuple[int, ...] = (1, 2, 4, 8)
    memory_margin_gb: float = 0.7
    eval_interval_steps: int = 500
    checkpoint_interval_steps: int = 500
    keep_last_checkpoints: int = 3
    gradient_clip_norm: float = 1.0
    dtype: str = "bfloat16"
    device: str = "cuda"
    compile_model: bool = True
    log_interval_steps: int = 10
    hardware_peak_flops: float = 88_000_000_000_000.0
    model_flops_per_token: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require(bool(self.output_dir.strip()), "output_dir must not be empty")
        _require(self.seed >= 0, "seed cannot be negative")
        _require(self.max_tokens >= 1, "max_tokens must be positive")
        _require(self.time_budget_hours > 0.0, "time_budget_hours must be positive")
        _require(self.global_batch_tokens >= 1, "global_batch_tokens must be positive")
        _require(self.micro_batch_sequences >= 0, "micro_batch_sequences cannot be negative")
        _require(
            bool(self.micro_batch_candidates)
            and all(value >= 1 for value in self.micro_batch_candidates)
            and tuple(sorted(set(self.micro_batch_candidates))) == self.micro_batch_candidates,
            "micro_batch_candidates must be unique, positive, and ascending",
        )
        if self.micro_batch_sequences:
            _require(
                self.micro_batch_sequences in self.micro_batch_candidates,
                "micro_batch_sequences must be zero or one of micro_batch_candidates",
            )
        _require(self.memory_margin_gb >= 0.0, "memory_margin_gb cannot be negative")
        for name in (
            "eval_interval_steps",
            "checkpoint_interval_steps",
            "keep_last_checkpoints",
            "log_interval_steps",
        ):
            _require(getattr(self, name) >= 1, f"{name} must be positive")
        _require(self.gradient_clip_norm > 0.0, "gradient_clip_norm must be positive")
        _require(self.hardware_peak_flops > 0.0, "hardware_peak_flops must be positive")
        _require(
            self.model_flops_per_token >= 0.0,
            "model_flops_per_token cannot be negative",
        )
        _require(self.dtype in {"bfloat16", "float32"}, "dtype must be bfloat16 or float32")
        _require(self.device in {"cuda", "cpu"}, "device must be cuda or cpu")


@dataclass(slots=True)
class RunState(YamlConfig):
    """Serializable progress state required for exact training resumption."""

    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    data_shard_index: int = 0
    data_offset: int = 0
    best_validation_loss: float = float("inf")
    last_validation_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    optimizer: str = "muon"
    completed: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("step", "tokens_seen", "epoch", "data_shard_index", "data_offset"):
            _require(getattr(self, name) >= 0, f"{name} cannot be negative")
        for name in ("best_validation_loss", "last_validation_loss"):
            value = getattr(self, name)
            _require(not math.isnan(value) and value >= 0.0, f"{name} must be non-negative")
        _require(self.elapsed_seconds >= 0.0, "elapsed_seconds cannot be negative")
        _require(self.optimizer in {"muon", "adamw"}, "optimizer must be muon or adamw")
