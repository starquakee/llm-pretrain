"""Optimizers and learning-rate schedules for pre-training.

The module deliberately depends only on PyTorch.  Configuration objects may be
dataclasses, simple namespaces, or mappings; this keeps the training core easy
to use from small experiments as well as from the project CLI.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _config_value(config: object, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_first(config: object, names: tuple[str, ...], default: Any) -> Any:
    sentinel = object()
    for name in names:
        value = _config_value(config, name, sentinel)
        if value is not sentinel:
            return value
    return default


@dataclass(frozen=True)
class ParameterPartition:
    """A complete, disjoint partition of trainable model parameters."""

    muon_names: tuple[str, ...]
    muon_parameters: tuple[nn.Parameter, ...]
    adamw_names: tuple[str, ...]
    adamw_parameters: tuple[nn.Parameter, ...]

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.muon_names + self.adamw_names


_OUTPUT_HEAD_NAMES = {"head", "lm_head", "output", "output_head", "classifier"}


def partition_parameters(model: nn.Module) -> ParameterPartition:
    """Put internal 2-D Linear weights in Muon and everything else in AdamW.

    Output heads are intentionally excluded even when they are implemented as a
    Linear layer.  Tied embeddings are also excluded because their parameter id
    is first discovered through an Embedding module.  The returned partition is
    validated to be both exhaustive and mutually exclusive.
    """

    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    name_by_id = {id(parameter): name for name, parameter in trainable.items()}
    muon_ids: set[int] = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        path_parts = {part.lower() for part in module_name.split(".")[:-1]}
        if not path_parts.intersection({"blocks", "layers", "h"}):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1].lower()
        if leaf_name in _OUTPUT_HEAD_NAMES:
            continue
        weight = module.weight
        if weight.requires_grad and weight.ndim == 2 and id(weight) in name_by_id:
            muon_ids.add(id(weight))

    muon_names: list[str] = []
    muon_parameters: list[nn.Parameter] = []
    adamw_names: list[str] = []
    adamw_parameters: list[nn.Parameter] = []
    for name, parameter in trainable.items():
        if id(parameter) in muon_ids:
            muon_names.append(name)
            muon_parameters.append(parameter)
        else:
            adamw_names.append(name)
            adamw_parameters.append(parameter)

    muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
    adamw_parameter_ids = {id(parameter) for parameter in adamw_parameters}
    all_parameter_ids = {id(parameter) for parameter in trainable.values()}
    if muon_parameter_ids & adamw_parameter_ids:
        raise RuntimeError("Muon and AdamW parameter groups overlap")
    if muon_parameter_ids | adamw_parameter_ids != all_parameter_ids:
        raise RuntimeError("Muon and AdamW parameter groups do not cover every trainable parameter")

    return ParameterPartition(
        muon_names=tuple(muon_names),
        muon_parameters=tuple(muon_parameters),
        adamw_names=tuple(adamw_names),
        adamw_parameters=tuple(adamw_parameters),
    )


def require_muon() -> Any:
    """Return the official PyTorch Muon optimizer or raise an actionable error."""

    muon = getattr(torch.optim, "Muon", None)
    if muon is None:
        raise RuntimeError(
            "torch.optim.Muon is unavailable. Install the locked PyTorch build that "
            "includes the official Muon optimizer, then rerun `llm-pretrain doctor`."
        )
    required_options = {"lr", "momentum", "weight_decay", "nesterov", "ns_steps", "adjust_lr_fn"}
    try:
        parameters = inspect.signature(muon).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to inspect torch.optim.Muon; the installed PyTorch build is unsupported"
        ) from exc
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        missing = sorted(required_options.difference(parameters))
        if missing:
            raise RuntimeError(
                "The installed torch.optim.Muon has an incompatible API; missing options: "
                + ", ".join(missing)
            )
    return muon


class OptimizerBundle:
    """Small common interface for the Muon+AdamW pair and AdamW baseline."""

    def __init__(self, optimizers: Mapping[str, torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("at least one optimizer is required")
        self.optimizers = dict(optimizers)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = set(self.optimizers)
        actual = set(state_dict)
        if expected != actual:
            raise ValueError(
                "optimizer checkpoint keys differ: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[name])


def create_optimizers(
    model: nn.Module,
    config: object,
    *,
    use_muon: bool | None = None,
) -> OptimizerBundle:
    """Create the Muon experiment optimizer pair or the AdamW baseline."""

    partition = partition_parameters(model)
    if use_muon is None:
        use_muon = _config_value(config, "optimizer", "muon") == "muon"
    if not use_muon:
        optimizer = torch.optim.AdamW(
            tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(_config_value(config, "adamw_lr", 6e-4)),
            betas=tuple(_config_value(config, "adamw_betas", (0.9, 0.95))),
            weight_decay=float(_config_value(config, "adamw_weight_decay", 0.1)),
        )
        return OptimizerBundle({"adamw": optimizer})

    if not partition.muon_parameters:
        raise ValueError("no eligible internal Linear weights were found for Muon")
    muon_type = require_muon()
    muon = muon_type(
        partition.muon_parameters,
        lr=float(_config_value(config, "muon_lr", 0.02)),
        momentum=float(_config_value(config, "muon_momentum", 0.95)),
        nesterov=bool(_config_value(config, "muon_nesterov", True)),
        ns_steps=int(_config_value(config, "muon_ns_steps", 5)),
        weight_decay=float(_config_value(config, "muon_weight_decay", 0.01)),
        adjust_lr_fn=_config_first(config, ("muon_lr_adjustment", "muon_adjust_lr_fn"), "original"),
    )
    auxiliary = torch.optim.AdamW(
        partition.adamw_parameters,
        lr=float(_config_first(config, ("auxiliary_lr", "aux_adamw_lr"), 3e-4)),
        betas=tuple(_config_first(config, ("auxiliary_betas", "aux_adamw_betas"), (0.9, 0.95))),
        weight_decay=float(
            _config_first(config, ("auxiliary_weight_decay", "aux_adamw_weight_decay"), 0.01)
        ),
    )
    return OptimizerBundle({"muon": muon, "adamw": auxiliary})


def warmup_cosine_factor(
    update_index: int,
    total_updates: int,
    warmup_updates: int,
    *,
    min_lr_ratio: float = 0.1,
) -> float:
    """LR factor for a zero-based optimizer update index."""

    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    update_index = min(max(update_index, 0), total_updates - 1)
    if warmup_updates and update_index < warmup_updates:
        return (update_index + 1) / warmup_updates
    decay_updates = max(total_updates - warmup_updates - 1, 1)
    progress = (update_index - warmup_updates) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class WarmupCosineScheduler:
    """A serializable scheduler that applies the same factor to a bundle."""

    def __init__(
        self,
        optimizer: OptimizerBundle | torch.optim.Optimizer,
        total_updates: int,
        warmup_updates: int,
        *,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.min_lr_ratio = min_lr_ratio
        self.update_index = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._apply()

    def _apply(self) -> None:
        factor = warmup_cosine_factor(
            self.update_index,
            self.total_updates,
            self.warmup_updates,
            min_lr_ratio=self.min_lr_ratio,
        )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.update_index = min(self.update_index + 1, self.total_updates - 1)
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "min_lr_ratio": self.min_lr_ratio,
            "update_index": self.update_index,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict["total_updates"]) != self.total_updates:
            raise ValueError("scheduler total_updates differs from checkpoint")
        if int(state_dict["warmup_updates"]) != self.warmup_updates:
            raise ValueError("scheduler warmup_updates differs from checkpoint")
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.update_index = int(state_dict["update_index"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler parameter-group count differs from checkpoint")
        self._apply()


def create_scheduler(
    optimizer: OptimizerBundle | torch.optim.Optimizer,
    total_updates: int,
    *,
    warmup_fraction: float = 0.02,
    min_lr_ratio: float = 0.1,
) -> WarmupCosineScheduler:
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    warmup_updates = min(int(total_updates * warmup_fraction), max(total_updates - 1, 0))
    return WarmupCosineScheduler(
        optimizer,
        total_updates,
        warmup_updates,
        min_lr_ratio=min_lr_ratio,
    )


def create_scheduler_from_config(
    optimizer: OptimizerBundle | torch.optim.Optimizer,
    total_updates: int,
    config: object,
) -> WarmupCosineScheduler:
    """Create the project schedule directly from ``OptimizerConfig``."""

    return create_scheduler(
        optimizer,
        total_updates,
        warmup_fraction=float(_config_value(config, "warmup_ratio", 0.02)),
        min_lr_ratio=float(_config_value(config, "min_lr_ratio", 0.1)),
    )


def optimizer_parameters(
    optimizer: OptimizerBundle | torch.optim.Optimizer,
) -> Iterable[nn.Parameter]:
    """Yield every parameter referenced by an optimizer, without duplicates."""

    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter
