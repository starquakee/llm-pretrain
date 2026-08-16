"""Atomic and bounded local checkpoints."""

from __future__ import annotations

import dataclasses
import os
import random
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        # Loading a checkpoint with ``map_location="cuda"`` also moves the
        # serialized CUDA RNG byte tensors onto the GPU.  The CUDA RNG API
        # deliberately accepts CPU ByteTensors, so normalise them before
        # restoring.  This keeps full-state resumes working regardless of the
        # checkpoint map location selected by the caller.
        cuda_states = [
            rng_state.detach().to(device="cpu", dtype=torch.uint8) for rng_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _run_state_dict(run_state: object | None) -> dict[str, Any]:
    if run_state is None:
        return {}
    if dataclasses.is_dataclass(run_state) and not isinstance(run_state, type):
        return dataclasses.asdict(run_state)
    if isinstance(run_state, Mapping):
        return dict(run_state)
    if hasattr(run_state, "state_dict"):
        stateful: Any = run_state
        return dict(stateful.state_dict())
    if hasattr(run_state, "__dict__"):
        return dict(vars(run_state))
    raise TypeError(
        "run_state must be a dataclass, mapping, state_dict object, or attribute object"
    )


def make_checkpoint_payload(
    model: nn.Module,
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    run_state: object | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "run_state": _run_state_dict(run_state),
        "rng_state": capture_rng_state(),
        "metrics": dict(metrics or {}),
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    run_state: object | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    payload = make_checkpoint_payload(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_state=run_state,
        metrics=metrics,
    )
    _atomic_torch_save(payload, destination)
    return destination


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older supported torch builds
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('format_version')!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def _restore_run_state(target: object, values: Mapping[str, Any]) -> None:
    if isinstance(target, MutableMapping):
        target.clear()
        target.update(values)
        return
    if hasattr(target, "load_state_dict"):
        stateful: Any = target
        stateful.load_state_dict(dict(values))
        return
    for name, value in values.items():
        if hasattr(target, name):
            setattr(target, name, value)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    run_state: object | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a full checkpoint and optionally restore each supplied component."""

    payload = _torch_load(Path(path), map_location)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if run_state is not None:
        _restore_run_state(run_state, payload["run_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def load_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    return load_checkpoint(
        path,
        model=model,
        map_location=map_location,
        restore_rng=False,
        strict=strict,
    )


class CheckpointManager:
    """Save atomic step checkpoints, retaining recent checkpoints plus best."""

    def __init__(self, directory: str | Path, *, keep_last: int = 3) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be at least one")
        self.directory = Path(directory)
        self.keep_last = keep_last

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def latest_path(self) -> Path | None:
        candidates = sorted(self.directory.glob("step_*.pt")) if self.directory.exists() else []
        return candidates[-1] if candidates else None

    def save(
        self,
        step: int,
        model: nn.Module,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        run_state: object | None = None,
        metrics: Mapping[str, Any] | None = None,
        is_best: bool = False,
        tag: str | None = None,
    ) -> Path:
        filename = f"{tag}.pt" if tag else f"step_{step:012d}.pt"
        path = self.directory / filename
        payload = make_checkpoint_payload(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_state=run_state,
            metrics=metrics,
        )
        _atomic_torch_save(payload, path)
        if is_best:
            _atomic_torch_save(payload, self.best_path)
        if tag is None:
            self._prune()
        return path

    def _prune(self) -> None:
        candidates = sorted(self.directory.glob("step_*.pt"))
        for stale_path in candidates[: -self.keep_last]:
            stale_path.unlink()
