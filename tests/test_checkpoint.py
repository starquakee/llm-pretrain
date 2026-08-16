from __future__ import annotations

import random

import pytest
import torch
from torch import nn

from llm_pretrain.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from llm_pretrain.config import RunState
from llm_pretrain.optim import create_optimizers, create_scheduler


def test_full_checkpoint_restores_model_optimizer_scheduler_state_and_rng(tmp_path) -> None:
    random.seed(17)
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    optimizer = create_optimizers(model, {"optimizer": "adamw"})
    scheduler = create_scheduler(optimizer, 5, warmup_fraction=0.2)
    state = RunState(step=2, tokens_seen=256, data_shard_index=3, data_offset=99)

    loss = model(torch.randn(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "resume.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_state=state,
        metrics={"validation/loss": 2.5},
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)

    for parameter in model.parameters():
        parameter.data.zero_()
    state.step = 0
    random.seed(999)
    torch.manual_seed(999)
    load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_state=state,
    )

    assert state.step == 2
    assert state.data_shard_index == 3
    assert state.data_offset == 99
    assert scheduler.update_index == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name])
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_checkpoint_manager_keeps_latest_three_and_independent_best(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=3)
    model = nn.Linear(2, 2)
    for step in range(1, 6):
        manager.save(step, model, run_state={"step": step}, is_best=step == 2)

    assert [path.name for path in sorted(tmp_path.glob("step_*.pt"))] == [
        "step_000000000003.pt",
        "step_000000000004.pt",
        "step_000000000005.pt",
    ]
    latest = manager.latest_path()
    assert latest is not None
    assert latest.name == "step_000000000005.pt"
    assert manager.best_path.exists()
    best = load_checkpoint(manager.best_path, restore_rng=False)
    assert best["run_state"]["step"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_checkpoint(path, restore_rng=False)
