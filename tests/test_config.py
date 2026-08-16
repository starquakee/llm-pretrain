from __future__ import annotations

import pytest

from llm_pretrain.config import (
    ConfigError,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RunState,
    TrainConfig,
)


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(),
        DataConfig(),
        OptimizerConfig(),
        TrainConfig(),
        RunState(),
    ],
)
def test_yaml_round_trip_is_lossless(tmp_path, config) -> None:
    destination = tmp_path / f"{type(config).__name__}.yaml"
    config.save_yaml(destination)

    restored = type(config).load_yaml(destination)

    assert restored == config


def test_yaml_loader_rejects_unknown_fields(tmp_path) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text("vocab_size: 256\nmade_up_setting: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="made_up_setting"):
        ModelConfig.load_yaml(source)


def test_yaml_loader_rejects_wrong_scalar_types(tmp_path) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text('n_layers: "12"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="n_layers"):
        ModelConfig.load_yaml(source)


def test_sequence_values_are_restored_as_tuples(tmp_path) -> None:
    source = tmp_path / "train.yaml"
    source.write_text("micro_batch_candidates: [1, 2, 4]\n", encoding="utf-8")

    config = TrainConfig.load_yaml(source)

    assert config.micro_batch_candidates == (1, 2, 4)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelConfig(d_model=63, n_heads=8), "divisible"),
        (lambda: ModelConfig(dropout=0.1), "dropout"),
        (lambda: DataConfig(english_web_ratio=0.2), "sum to 1"),
        (lambda: OptimizerConfig(muon_lr=0.0), "muon_lr"),
        (lambda: TrainConfig(micro_batch_candidates=(1, 4, 2)), "ascending"),
        (lambda: RunState(tokens_seen=-1), "tokens_seen"),
    ],
)
def test_invalid_programmatic_configuration_is_rejected(factory, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        factory()


def test_default_architecture_contract() -> None:
    config = ModelConfig()

    assert config.vocab_size == 24_576
    assert config.n_layers == 12
    assert config.d_model == 768
    assert config.n_heads == 12
    assert config.head_dim == 64
    assert config.intermediate_size == 1_920
    assert config.bias is False
    assert config.dropout == 0.0
