from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from llm_pretrain.optim import (
    create_optimizers,
    create_scheduler,
    partition_parameters,
    require_muon,
    warmup_cosine_factor,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(inputs))


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.final_norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def test_parameter_partition_is_exhaustive_disjoint_and_only_block_linear_weights() -> None:
    model = TinyLM()
    partition = partition_parameters(model)

    assert partition.muon_names == ("blocks.0.proj.weight", "blocks.1.proj.weight")
    assert "token_embedding.weight" in partition.adamw_names
    assert "blocks.0.proj.bias" in partition.adamw_names
    assert "blocks.0.norm.weight" in partition.adamw_names
    assert set(partition.all_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    muon_ids = {id(value) for value in partition.muon_parameters}
    adamw_ids = {id(value) for value in partition.adamw_parameters}
    assert not (muon_ids & adamw_ids)


def test_missing_muon_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    with pytest.raises(RuntimeError, match=r"torch\.optim\.Muon.*doctor"):
        require_muon()


def test_create_muon_pair_uses_project_config_names(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeMuon(torch.optim.SGD):
        def __init__(
            self,
            params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
            lr: float,
            momentum: float,
            weight_decay: float,
            nesterov: bool,
            ns_steps: int,
            adjust_lr_fn: str,
        ) -> None:
            created.update(ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
            super().__init__(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )

    monkeypatch.setattr(torch.optim, "Muon", FakeMuon, raising=False)
    config = SimpleNamespace(
        optimizer="muon",
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_weight_decay=0.01,
        muon_lr_adjustment="original",
        auxiliary_lr=3e-4,
        auxiliary_betas=(0.9, 0.95),
        auxiliary_weight_decay=0.01,
    )
    bundle = create_optimizers(TinyLM(), config)

    assert set(bundle.optimizers) == {"muon", "adamw"}
    assert created == {"ns_steps": 5, "adjust_lr_fn": "original"}
    assert bundle.optimizers["adamw"].param_groups[0]["lr"] == pytest.approx(3e-4)


def test_adamw_baseline_covers_every_parameter() -> None:
    model = TinyLM()
    bundle = create_optimizers(model, SimpleNamespace(optimizer="adamw"))
    optimized = {id(parameter) for group in bundle.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}


def test_warmup_cosine_and_scheduler_resume() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = create_scheduler(
        optimizer,
        10,
        warmup_fraction=0.2,
        min_lr_ratio=0.1,
    )
    assert scheduler.get_last_lr() == pytest.approx([0.5])
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.0])
    for _ in range(8):
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.1])

    restored_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=1.0)
    restored = create_scheduler(restored_optimizer, 10, warmup_fraction=0.2, min_lr_ratio=0.1)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert warmup_cosine_factor(9, 10, 2, min_lr_ratio=0.1) == pytest.approx(0.1)
