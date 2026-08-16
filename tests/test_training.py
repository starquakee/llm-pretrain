from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from llm_pretrain.checkpoint import CheckpointManager
from llm_pretrain.optim import create_optimizers, create_scheduler
from llm_pretrain.training import (
    ABRunMetrics,
    LocalMetricLogger,
    compute_gradient_accumulation_steps,
    evaluate_ab_gate,
    pending_muon_calibrations,
    probe_micro_batch_sequences,
    train,
)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 13) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.head(self.embedding(input_ids))
        if labels is None:
            return {"logits": logits}
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
        )
        return {"logits": logits, "loss": loss}


def _config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "global_batch_tokens": 8,
        "sequence_length": 4,
        "micro_batch_sequences": 1,
        "max_steps": 2,
        "checkpoint_interval_steps": 1,
        "eval_interval_steps": 1,
        "validation_batches": 1,
        "gradient_clip_norm": 1.0,
        "dtype": "float32",
        "compile_model": False,
        "log_interval_steps": 1,
    }
    values.update(overrides)
    return values


def test_cpu_training_accumulates_to_global_batch_and_checkpoints(tmp_path) -> None:
    torch.manual_seed(3)
    model = TinyCausalLM()
    optimizer = create_optimizers(model, {"optimizer": "adamw", "adamw_lr": 0.01})
    scheduler = create_scheduler(optimizer, 2, warmup_fraction=0.0)
    batches = [torch.randint(0, 13, (1, 4)), torch.randint(0, 13, (1, 4))]
    manager = CheckpointManager(tmp_path / "checkpoints")
    log_path = tmp_path / "metrics.jsonl"

    with LocalMetricLogger(log_path) as logger:
        result = train(
            model,
            batches,
            optimizer,
            _config(),
            scheduler=scheduler,
            validation_loader=batches[:1],
            checkpoint_manager=manager,
            metric_logger=logger,
            device="cpu",
        )

    assert result.stopped_reason == "completed"
    state = result.run_state
    assert hasattr(state, "step")
    assert state.step == 2
    assert state.tokens_seen == 16
    assert state.data_cursor == 16
    assert result.best_checkpoint == manager.best_path
    assert result.last_checkpoint is not None
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any("train/tokens_per_second" in event for event in events)
    assert any("validation/loss" in event for event in events)


class NonFiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        del input_ids, labels
        return self.weight * torch.tensor(float("nan"))


def test_non_finite_loss_stops_without_optimizer_step_and_writes_emergency(tmp_path) -> None:
    model = NonFiniteModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = model.weight.detach().clone()
    result = train(
        model,
        [torch.ones((1, 4), dtype=torch.long)],
        optimizer,
        _config(max_steps=1, global_batch_tokens=4),
        checkpoint_manager=CheckpointManager(tmp_path),
        device="cpu",
    )
    assert result.stopped_reason == "non_finite_loss"
    state = result.run_state
    assert hasattr(state, "step")
    assert state.step == 0
    torch.testing.assert_close(model.weight, initial)
    assert result.last_checkpoint == tmp_path / "emergency.pt"
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.exists()


def test_ab_gate_and_single_calibration_sweep_are_pure() -> None:
    adamw = ABRunMetrics((2.2, 2.0, 1.8), tokens_per_second=100.0)
    passing_muon = ABRunMetrics((2.21, 2.01, 1.81), tokens_per_second=85.0)
    passed = evaluate_ab_gate(passing_muon, adamw)
    assert passed.passed
    assert pending_muon_calibrations(passed) == ()

    failing_muon = ABRunMetrics((2.4, 2.3, 2.2), tokens_per_second=70.0)
    failed = evaluate_ab_gate(failing_muon, adamw)
    assert not failed.passed
    assert pending_muon_calibrations(failed) == (0.01, 0.04)
    assert pending_muon_calibrations(failed, attempted_lrs=(0.01, 0.04)) == ()


def test_batch_math_and_cpu_probe() -> None:
    assert compute_gradient_accumulation_steps(131_072, 2, 1024) == 64
    with pytest.raises(ValueError, match="exactly divisible"):
        compute_gradient_accumulation_steps(10, 1, 4)
    assert (
        probe_micro_batch_sequences(
            TinyCausalLM(), sequence_length=4, candidates=(1, 2, 4), device="cpu"
        )
        == 4
    )
