"""Single-device pre-training loop and deterministic A/B gate logic."""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checkpoint import CheckpointManager, capture_rng_state, restore_rng_state
from .optim import OptimizerBundle, WarmupCosineScheduler


def read_nvidia_temperature_c(device_index: int = 0) -> float:
    """Read one NVIDIA GPU temperature, returning NaN when telemetry is unavailable."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            return math.nan
        return float(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
        return math.nan


def _config_value(config: object, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _state_get(state: object, name: str, default: Any) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _state_set(state: object, name: str, value: Any) -> None:
    if isinstance(state, dict):
        state[name] = value
    elif hasattr(state, name):
        setattr(state, name, value)


def _state_set_first(state: object, names: tuple[str, ...], value: Any) -> None:
    if isinstance(state, dict):
        state[names[0]] = value
        return
    for name in names:
        if hasattr(state, name):
            setattr(state, name, value)
            return


@dataclass
class TrainingRunState:
    step: int = 0
    tokens_seen: int = 0
    data_cursor: int = 0
    best_val_loss: float = math.inf
    validation_losses: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TrainResult:
    run_state: Any
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    stopped_reason: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ABRunMetrics:
    validation_losses: tuple[float, ...]
    tokens_per_second: float
    finite: bool = True


@dataclass(frozen=True)
class ABGateDecision:
    passed: bool
    muon_mean_loss: float
    adamw_mean_loss: float
    throughput_ratio: float
    reasons: tuple[str, ...]


def evaluate_ab_gate(
    muon: ABRunMetrics,
    adamw: ABRunMetrics,
    *,
    loss_tolerance: float = 0.05,
    minimum_throughput_ratio: float = 0.8,
) -> ABGateDecision:
    """Pure decision function for the 100M-token Muon/AdamW comparison."""

    if len(muon.validation_losses) < 3 or len(adamw.validation_losses) < 3:
        raise ValueError("both A/B runs require at least three validation losses")
    muon_mean = sum(muon.validation_losses[-3:]) / 3
    adamw_mean = sum(adamw.validation_losses[-3:]) / 3
    throughput_ratio = (
        muon.tokens_per_second / adamw.tokens_per_second if adamw.tokens_per_second > 0 else 0.0
    )
    reasons: list[str] = []
    if not muon.finite or not math.isfinite(muon_mean):
        reasons.append("Muon produced a non-finite result")
    if not adamw.finite or not math.isfinite(adamw_mean):
        reasons.append("AdamW baseline produced a non-finite result")
    if (
        math.isfinite(muon_mean)
        and math.isfinite(adamw_mean)
        and muon_mean > adamw_mean + loss_tolerance
    ):
        reasons.append("Muon validation loss exceeds the allowed tolerance")
    if throughput_ratio < minimum_throughput_ratio:
        reasons.append("Muon throughput is below the required ratio")
    return ABGateDecision(
        passed=not reasons,
        muon_mean_loss=muon_mean,
        adamw_mean_loss=adamw_mean,
        throughput_ratio=throughput_ratio,
        reasons=tuple(reasons),
    )


def pending_muon_calibrations(
    decision: ABGateDecision,
    attempted_lrs: Iterable[float] = (),
    *,
    calibration_lrs: Sequence[float] = (0.01, 0.04),
) -> tuple[float, ...]:
    """Return the one allowed calibration sweep, or no work after it is spent."""

    if decision.passed:
        return ()
    attempted = {round(float(lr), 12) for lr in attempted_lrs}
    return tuple(float(lr) for lr in calibration_lrs if round(float(lr), 12) not in attempted)


def compute_gradient_accumulation_steps(
    global_batch_tokens: int,
    micro_batch_sequences: int,
    sequence_length: int,
) -> int:
    micro_batch_tokens = micro_batch_sequences * sequence_length
    if global_batch_tokens <= 0 or micro_batch_tokens <= 0:
        raise ValueError("batch sizes and sequence length must be positive")
    if global_batch_tokens % micro_batch_tokens:
        raise ValueError(
            "global_batch_tokens must be exactly divisible by "
            "micro_batch_sequences * sequence_length"
        )
    return global_batch_tokens // micro_batch_tokens


def probe_micro_batch_sequences(
    model: nn.Module,
    *,
    sequence_length: int,
    candidates: Sequence[int] = (1, 2, 4, 8),
    device: str | torch.device = "cuda",
    memory_margin_gb: float = 0.7,
    use_bf16: bool = True,
) -> int:
    """Select the largest candidate that completes backward with the margin.

    The probe changes no weights and restores CPU/CUDA RNG state.  It is exposed
    separately because a DataLoader must normally be constructed *after* its
    batch size is known.
    """

    target_device = torch.device(device)
    ordered = tuple(sorted(set(int(value) for value in candidates)))
    if not ordered or ordered[0] < 1:
        raise ValueError("micro-batch candidates must be positive")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if target_device.type != "cuda":
        return ordered[-1]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA micro-batch probing requested, but CUDA is unavailable")

    model.to(target_device)
    was_training = model.training
    rng_state = capture_rng_state()
    selected: int | None = None
    required_free_bytes = int(memory_margin_gb * 1024**3)
    try:
        sample: Tensor | None = None
        for candidate in ordered:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            try:
                sample = torch.zeros(
                    (candidate, sequence_length), dtype=torch.long, device=target_device
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = _forward_loss(model, sample, sample)
                loss.backward()
                free_bytes, _ = torch.cuda.mem_get_info(target_device)
                if free_bytes >= required_free_bytes:
                    selected = candidate
                else:
                    break
            except RuntimeError as error:
                if not _is_oom(error):
                    raise
                break
            finally:
                model.zero_grad(set_to_none=True)
                if sample is not None:
                    sample = None
                torch.cuda.empty_cache()
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)
    if selected is None:
        raise RuntimeError(
            "no micro-batch candidate leaves the configured CUDA memory margin; "
            "close GPU applications or reduce sequence length"
        )
    return selected


class LocalMetricLogger:
    """Append-only JSONL metrics with optional TensorBoard mirroring."""

    def __init__(
        self,
        jsonl_path: str | Path | None,
        tensorboard_dir: str | Path | None = None,
    ) -> None:
        self._handle = None
        self._writer = None
        if jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise RuntimeError(
                    "TensorBoard logging requested, but tensorboard is not installed"
                ) from exc
            self._writer = SummaryWriter(str(tensorboard_dir))

    def log(self, step: int, values: Mapping[str, Any]) -> None:
        event = {"step": step, "time": time.time(), **dict(values)}
        if self._handle is not None:
            self._handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            self._handle.flush()
        if self._writer is not None:
            for name, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self._writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> LocalMetricLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _batch_to_device(
    batch: object, device: torch.device
) -> tuple[Tensor, Tensor, int, object | None]:
    input_ids: Tensor
    labels: Tensor | None
    cursor: object | None = None
    if isinstance(batch, Tensor):
        input_ids = batch
        labels = None
    elif isinstance(batch, Mapping):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        cursor = batch.get("cursor")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        input_ids, labels = batch
    else:
        raise TypeError("batch must be a Tensor, (input_ids, labels), or mapping")
    if not isinstance(input_ids, Tensor):
        try:
            input_ids = torch.as_tensor(input_ids)
        except (TypeError, ValueError) as exc:
            raise TypeError("input_ids must be convertible to a torch tensor") from exc
    if labels is not None and not isinstance(labels, Tensor):
        try:
            labels = torch.as_tensor(labels)
        except (TypeError, ValueError) as exc:
            raise TypeError("labels must be convertible to a torch tensor") from exc
    nominal_tokens = input_ids.numel()
    input_ids = input_ids.to(device, non_blocking=True)
    if labels is None:
        if input_ids.ndim < 2 or input_ids.shape[-1] < 2:
            raise ValueError("an unlabeled packed batch needs a sequence dimension of at least two")
        labels = input_ids
    else:
        labels = labels.to(device, non_blocking=True)
    return input_ids, labels, nominal_tokens, cursor


def _sync_run_state_cursor(state: object, cursor: object | None, fallback_tokens: int) -> None:
    if cursor is None:
        name = "data_offset" if hasattr(state, "data_offset") else "data_cursor"
        _state_set(state, name, int(_state_get(state, name, 0)) + fallback_tokens)
        return

    def cursor_value(name: str, default: int) -> int:
        if isinstance(cursor, Mapping):
            return int(cursor.get(name, default))
        return int(getattr(cursor, name, default))

    _state_set_first(state, ("epoch",), cursor_value("epoch", _state_get(state, "epoch", 0)))
    _state_set_first(
        state,
        ("data_shard_index",),
        cursor_value("shard_position", _state_get(state, "data_shard_index", 0)),
    )
    _state_set_first(
        state,
        ("data_offset", "data_cursor"),
        cursor_value("token_offset", _state_get(state, "data_offset", 0)),
    )


def _forward_loss(model: nn.Module, input_ids: Tensor, labels: Tensor) -> Tensor:
    signature = inspect.signature(model.forward)
    if "labels" in signature.parameters:
        output = model(input_ids, labels=labels)
    elif "targets" in signature.parameters:
        output = model(input_ids, targets=labels)
    else:
        output = model(input_ids)

    if isinstance(output, Tensor):
        if output.ndim == 0:
            return output
        logits = output
    elif isinstance(output, Mapping):
        loss = output.get("loss")
        if isinstance(loss, Tensor):
            return loss
        logits = output.get("logits")
    elif hasattr(output, "loss") and isinstance(output.loss, Tensor):
        return output.loss
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, (tuple, list)):
        scalar = next(
            (item for item in output if isinstance(item, Tensor) and item.ndim == 0),
            None,
        )
        if scalar is not None:
            return scalar
        logits = output[0]
    else:
        raise TypeError("model output must contain loss or logits")
    if not isinstance(logits, Tensor):
        raise TypeError("model output logits are not a Tensor")
    if logits.ndim >= 3 and logits.shape[:-1] == labels.shape:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


def _next_batch(
    iterator: Iterator[object], source: Iterable[object]
) -> tuple[object, Iterator[object]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iterator = iter(source)
        if new_iterator is iterator:
            raise RuntimeError("one-shot training iterator was exhausted") from None
        try:
            return next(new_iterator), new_iterator
        except StopIteration:
            raise RuntimeError("training loader is empty") from None


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable[object],
    *,
    device: torch.device,
    max_batches: int | None = None,
    use_bf16: bool = True,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    autocast = torch.autocast if device.type in {"cuda", "cpu"} else None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids, labels, _, _ = _batch_to_device(batch, device)
        context = (
            autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)
            if autocast is not None
            else contextlib.nullcontext()
        )
        with context:
            loss = _forward_loss(model, input_ids, labels)
        losses.append(float(loss.detach()))
    model.train(was_training)
    if not losses:
        raise ValueError("validation loader yielded no batches")
    return sum(losses) / len(losses)


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def _save_emergency(
    manager: CheckpointManager | None,
    model: nn.Module,
    optimizer: object,
    scheduler: object | None,
    run_state: object,
    reason: str,
) -> Path | None:
    if manager is None:
        return None
    try:
        return manager.save(
            int(_state_get(run_state, "step", 0)),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_state=run_state,
            metrics={"stop_reason": reason},
            tag="emergency",
        )
    except BaseException:
        return None


def train(
    model: nn.Module,
    train_loader: Iterable[object],
    optimizer: OptimizerBundle | torch.optim.Optimizer,
    config: object,
    *,
    scheduler: WarmupCosineScheduler | None = None,
    validation_loader: Iterable[object] | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    metric_logger: LocalMetricLogger | None = None,
    run_state: object | None = None,
    device: str | torch.device | None = None,
    temperature_reader: Any | None = None,
) -> TrainResult:
    """Train until the configured token/update target, stopping safely on failure.

    OOM and non-finite loss/gradients create a best-effort ``emergency.pt`` and
    return a stopped result.  The function never changes the micro-batch size.
    """

    state = run_state if run_state is not None else TrainingRunState()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device or _config_value(config, "device", default_device))
    model.to(target_device)
    model.train()
    forward_model = model
    if bool(_config_value(config, "compile_model", False)):
        compile_function = getattr(torch, "compile", None)
        if compile_function is None:
            raise RuntimeError("compile_model=true, but this PyTorch build has no torch.compile")
        forward_model = compile_function(model)
    global_batch_tokens = int(_config_value(config, "global_batch_tokens", 131_072))
    model_config = getattr(model, "config", None)
    model_sequence_length = getattr(model_config, "max_seq_len", 1024)
    sequence_length = int(_config_value(config, "sequence_length", model_sequence_length))
    micro_batch_sequences = int(_config_value(config, "micro_batch_sequences", 1))
    if micro_batch_sequences == 0:
        raise ValueError(
            "micro_batch_sequences is 0; call probe_micro_batch_sequences before "
            "constructing the training DataLoader"
        )
    accumulation_steps = compute_gradient_accumulation_steps(
        global_batch_tokens, micro_batch_sequences, sequence_length
    )
    max_tokens = int(_config_value(config, "max_tokens", _config_value(config, "target_tokens", 0)))
    max_steps = int(_config_value(config, "max_steps", 0))
    if max_steps <= 0:
        if max_tokens <= 0:
            raise ValueError("config must specify positive max_tokens/target_tokens or max_steps")
        max_steps = math.ceil(max_tokens / global_batch_tokens)
    checkpoint_interval = int(
        _config_value(
            config,
            "checkpoint_interval_steps",
            _config_value(config, "checkpoint_interval", 500),
        )
    )
    validation_interval = int(
        _config_value(
            config,
            "eval_interval_steps",
            _config_value(config, "validation_interval", checkpoint_interval),
        )
    )
    validation_batches = _config_value(config, "validation_batches", None)
    grad_clip = float(
        _config_value(config, "gradient_clip_norm", _config_value(config, "grad_clip", 1.0))
    )
    use_bf16 = _config_value(config, "dtype", "bfloat16") == "bfloat16"
    log_interval = int(_config_value(config, "log_interval_steps", 1))
    time_budget_seconds = float(_config_value(config, "time_budget_hours", math.inf)) * 3600
    hardware_peak_flops = float(_config_value(config, "hardware_peak_flops", 0.0))
    model_flops_per_token = float(_config_value(config, "model_flops_per_token", 0.0))
    if model_flops_per_token <= 0:
        model_flops_per_token = 6.0 * sum(parameter.numel() for parameter in model.parameters())

    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint: Path | None = None
    best_checkpoint: Path | None = None
    if checkpoint_manager is not None and checkpoint_manager.best_path.exists():
        best_checkpoint = checkpoint_manager.best_path
    last_metrics: dict[str, float] = {}
    stopped_reason = "completed"

    while int(_state_get(state, "step", 0)) < max_steps:
        if float(_state_get(state, "elapsed_seconds", 0.0)) >= time_budget_seconds:
            stopped_reason = "time_budget"
            break
        started = time.perf_counter()
        accumulated_loss = 0.0
        actual_tokens = 0
        try:
            for _ in range(accumulation_steps):
                batch, iterator = _next_batch(iterator, train_loader)
                input_ids, labels, batch_tokens, cursor = _batch_to_device(batch, target_device)
                if batch_tokens != micro_batch_sequences * sequence_length:
                    raise ValueError(
                        "loader batch token count differs from configured "
                        "micro_batch_sequences * sequence_length"
                    )
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16 and target_device.type in {"cuda", "cpu"},
                ):
                    loss = _forward_loss(forward_model, input_ids, labels)
                if not torch.isfinite(loss):
                    stopped_reason = "non_finite_loss"
                    raise FloatingPointError(stopped_reason)
                (loss / accumulation_steps).backward()
                accumulated_loss += float(loss.detach())
                actual_tokens += batch_tokens
                _sync_run_state_cursor(state, cursor, batch_tokens)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                stopped_reason = "non_finite_gradients"
                raise FloatingPointError(stopped_reason)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        except FloatingPointError:
            optimizer.zero_grad(set_to_none=True)
            last_checkpoint = _save_emergency(
                checkpoint_manager, model, optimizer, scheduler, state, stopped_reason
            )
            break
        except RuntimeError as error:
            if not _is_oom(error):
                raise
            stopped_reason = "out_of_memory"
            optimizer.zero_grad(set_to_none=True)
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
            last_checkpoint = _save_emergency(
                checkpoint_manager, model, optimizer, scheduler, state, stopped_reason
            )
            break

        elapsed = max(time.perf_counter() - started, 1e-12)
        step = int(_state_get(state, "step", 0)) + 1
        tokens_seen = int(_state_get(state, "tokens_seen", 0)) + actual_tokens
        _state_set(state, "step", step)
        _state_set(state, "tokens_seen", tokens_seen)
        _state_set_first(
            state,
            ("elapsed_seconds",),
            float(_state_get(state, "elapsed_seconds", 0.0)) + elapsed,
        )
        peak_memory = (
            float(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0.0
        )
        last_metrics = {
            "train/loss": accumulated_loss / accumulation_steps,
            "train/grad_norm": float(grad_norm),
            "train/tokens_per_second": actual_tokens / elapsed,
            "train/peak_memory_bytes": peak_memory,
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/tokens_seen": float(tokens_seen),
        }
        if hardware_peak_flops > 0 and model_flops_per_token > 0:
            last_metrics["train/mfu"] = min(
                1.0,
                actual_tokens / elapsed * model_flops_per_token / hardware_peak_flops,
            )
        should_log = step % log_interval == 0 or step == max_steps
        if temperature_reader is not None and should_log:
            temperature = float(temperature_reader())
            if math.isfinite(temperature):
                last_metrics["system/gpu_temperature_c"] = temperature
        if metric_logger is not None and should_log:
            metric_logger.log(step, last_metrics)

        is_validation_step = validation_loader is not None and (
            step % validation_interval == 0 or step == max_steps
        )
        is_best = False
        if is_validation_step:
            assert validation_loader is not None
            val_loss = evaluate_loss(
                forward_model,
                validation_loader,
                device=target_device,
                max_batches=validation_batches,
                use_bf16=use_bf16,
            )
            if not math.isfinite(val_loss):
                stopped_reason = "non_finite_validation"
                last_checkpoint = _save_emergency(
                    checkpoint_manager, model, optimizer, scheduler, state, stopped_reason
                )
                break
            validation_losses = list(_state_get(state, "validation_losses", []))
            validation_losses.append(val_loss)
            _state_set(state, "validation_losses", validation_losses)
            best_val_loss = float(
                _state_get(
                    state,
                    "best_validation_loss",
                    _state_get(state, "best_val_loss", math.inf),
                )
            )
            is_best = val_loss < best_val_loss
            if is_best:
                _state_set_first(state, ("best_validation_loss", "best_val_loss"), val_loss)
            _state_set_first(state, ("last_validation_loss",), val_loss)
            last_metrics["validation/loss"] = val_loss
            if metric_logger is not None:
                metric_logger.log(step, {"validation/loss": val_loss})

        if checkpoint_manager is not None and (
            step % checkpoint_interval == 0 or step == max_steps or is_best
        ):
            last_checkpoint = checkpoint_manager.save(
                step,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_state=state,
                metrics=last_metrics,
                is_best=is_best,
            )
            if is_best:
                best_checkpoint = checkpoint_manager.best_path

    _state_set_first(state, ("completed",), stopped_reason == "completed")
    return TrainResult(
        run_state=state,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        stopped_reason=stopped_reason,
        metrics=last_metrics,
    )
