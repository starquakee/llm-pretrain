from __future__ import annotations

from pathlib import Path


def test_pipeline_orders_gate_before_formal_pretraining() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_training_pipeline.sh").read_text(
        encoding="utf-8"
    )
    gate = script.index('run_stage train-ab "$CLI" train ab')
    pretrain = script.index('run_stage train-pretrain "$CLI" train pretrain')
    assert gate < pretrain
    assert '--gate-record "$GATE_RECORD"' in script
    assert "--unsafe" not in script


def test_pipeline_keeps_every_cache_under_artifact_root() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "run_training_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert 'HF_HOME="$ARTIFACT_ROOT/cache/huggingface"' in script
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in script
    assert 'XDG_CACHE_HOME="$ARTIFACT_ROOT/cache/xdg"' in script
    assert 'TORCHINDUCTOR_CACHE_DIR="$ARTIFACT_ROOT/cache/torchinductor"' in script
