from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llm_pretrain import cli
from llm_pretrain.config import ConfigError
from llm_pretrain.data import DEFAULT_SOURCES, SourceManifest


def _write_pretrain_config(path: Path, artifact_root: Path) -> Path:
    payload = {
        "model": {
            "vocab_size": 256,
            "max_seq_len": 8,
            "n_layers": 1,
            "d_model": 16,
            "n_heads": 2,
            "intermediate_size": 32,
            "activation_checkpointing": False,
        },
        "data": {
            "data_dir": str(artifact_root / "data"),
            "sequence_length": 8,
            "tokenizer_vocab_size": 256,
            "validation_tokens": 8,
            "shard_size_tokens": 9,
            "num_workers": 0,
        },
        "optimizer": {"optimizer": "muon"},
        "train": {
            "output_dir": str(artifact_root / "runs" / "test"),
            "max_tokens": 32,
            "global_batch_tokens": 8,
            "micro_batch_sequences": 1,
            "micro_batch_candidates": [1],
            "device": "cpu",
            "dtype": "float32",
            "compile_model": False,
            "eval_interval_steps": 1,
            "checkpoint_interval_steps": 1,
            "log_interval_steps": 1,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _passing_gate(path: Path, *, throughput: float = 10.0) -> Path:
    run = {
        "validation_losses": [2.0, 1.9, 1.8],
        "tokens_per_second": throughput,
        "finite": True,
    }
    payload = {
        "schema_version": 1,
        "passed": True,
        "decision": {"passed": True, "reasons": []},
        "target_tokens_per_run": 100_000_000,
        "runs": {"muon": run, "adamw": run},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parser_exposes_complete_command_tree() -> None:
    parser = cli.build_parser()

    for argv in (
        ["doctor"],
        ["data", "prepare"],
        ["data", "tokenize"],
        ["tokenizer", "train"],
        ["tokenizer", "eval"],
        ["train", "ab", "--config", "x.yaml"],
        ["train", "pretrain", "--config", "x.yaml", "--gate-record", "gate.json"],
        ["train", "sft", "--config", "sft.yaml"],
        ["evaluate", "--config", "x.yaml"],
        ["generate", "--config", "x.yaml", "--prompt", "你好"],
    ):
        assert callable(parser.parse_args(argv).handler)


def test_help_does_not_import_or_start_training(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["train", "pretrain", "--help"])

    assert exc_info.value.code == 0
    assert "requires passing A/B gate" in capsys.readouterr().out


def test_nested_pretrain_config_loads_four_strict_types(tmp_path) -> None:
    configs = cli.load_pretrain_config(
        _write_pretrain_config(tmp_path / "pretrain.yaml", tmp_path / "artifacts")
    )

    assert configs.model.d_model == 16
    assert configs.data.sequence_length == 8
    assert configs.optimizer.optimizer == "muon"
    assert configs.train.device == "cpu"


def test_nested_pretrain_config_rejects_unknown_section(tmp_path) -> None:
    path = _write_pretrain_config(tmp_path / "pretrain.yaml", tmp_path / "artifacts")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["made_up"] = {}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="made_up"):
        cli.load_pretrain_config(path)


def test_nested_pretrain_config_rejects_architecture_data_mismatch(tmp_path) -> None:
    path = _write_pretrain_config(tmp_path / "pretrain.yaml", tmp_path / "artifacts")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["sequence_length"] = 16
    payload["data"]["shard_size_tokens"] = 17
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="max_seq_len"):
        cli.load_pretrain_config(path)


def test_data_prepare_is_offline_by_default(tmp_path, capsys) -> None:
    result = cli.main(["--artifact-root", str(tmp_path / "artifacts"), "data", "prepare"])

    assert result == 2
    assert "offline by default" in capsys.readouterr().err


def test_data_prepare_accepts_a_small_local_fixture(tmp_path, capsys) -> None:
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"text": text, "source": "fixture"}, ensure_ascii=False)
            for text in ("中文测试。", "另一个文档。")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "prepared"

    result = cli.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "data",
            "prepare",
            "--input",
            f"fixture={source}",
            "--output-dir",
            str(output),
            "--validation-fraction",
            "0",
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert (output / "train.jsonl").is_file()
    assert (output / "sources.json").is_file()
    assert not (output / "downloads").exists()


def test_network_source_byte_limiter_includes_boundary_document() -> None:
    documents = [
        cli.Document("abc", "source", "1"),
        cli.Document("中文", "source", "2"),
        cli.Document("unused", "source", "3"),
    ]

    limited = list(cli._limit_documents_by_utf8_bytes(documents, 5))

    assert [document.document_id for document in limited] == ["1", "2"]


def test_tokenizer_train_extracts_bounded_text_only_balanced_corpus(
    tmp_path, monkeypatch, capsys
) -> None:
    artifact_root = tmp_path / "artifacts"
    prepared = artifact_root / "data" / "prepared"
    prepared.mkdir(parents=True)
    sources = tuple(
        replace(source, revision=("a" * 40 if source.revision == "main" else source.revision))
        for source in DEFAULT_SOURCES
    )
    SourceManifest(sources=sources, seed=1337).write(prepared / "sources.json")
    rows = [
        {"id": "secret-ultra", "sha256": "u", "source": sources[0].name, "text": "中文网页"},
        {"id": "secret-wiki", "sha256": "w", "source": sources[1].name, "text": "百科"},
        {"id": "secret-en", "sha256": "e", "source": sources[2].name, "text": "hello"},
    ]
    (prepared / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    config = tmp_path / "tokenizer.yaml"
    config.write_text(
        yaml.safe_dump({"vocab_size": 263, "training_corpus_bytes": 100}), encoding="utf-8"
    )
    captured: dict[str, str] = {}

    def fake_train(input_files, model_prefix, **kwargs):
        sample = Path(input_files[0])
        captured["sample"] = sample.read_text(encoding="utf-8")
        model = Path(model_prefix).with_suffix(".model")
        vocab = Path(model_prefix).with_suffix(".vocab")
        model.write_bytes(b"model")
        vocab.write_text("vocab", encoding="utf-8")
        return model, vocab

    monkeypatch.setattr(cli, "train_sentencepiece", fake_train)
    result = cli.main(
        [
            "--artifact-root",
            str(artifact_root),
            "tokenizer",
            "train",
            "--config",
            str(config),
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert captured["sample"] == "中文网页\n百科\nhello\n"
    assert "secret-" not in captured["sample"]
    assert '\"source\"' not in captured["sample"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(passed=False), "blocks formal pretraining"),
        (lambda value: value.update(schema_version=999), "schema_version"),
        (
            lambda value: value["runs"]["muon"].update(validation_losses=[1.0]),
            "at least three",
        ),
        (lambda value: value["runs"]["muon"].update(finite=False), "non-finite"),
    ],
)
def test_gate_record_rejects_invalid_or_failed_runs(tmp_path, mutation, message) -> None:
    path = _passing_gate(tmp_path / "gate.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli.validate_gate_record(path)


def test_formal_budget_is_capped_and_aligned_to_global_batch() -> None:
    assert (
        cli.calculate_formal_token_budget(
            1_000.0,
            time_budget_hours=1.0,
            maximum_tokens=2_000_000,
            global_batch_tokens=131_072,
        )
        == 1_966_080
    )
    assert (
        cli.calculate_formal_token_budget(
            1_000_000.0,
            time_budget_hours=120.0,
            maximum_tokens=2_000_000_000,
            global_batch_tokens=131_072,
            requested_max_tokens=1_000_000,
        )
        == 917_504
    )


def test_pretrain_checks_gate_before_doctor_or_model_creation(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _write_pretrain_config(tmp_path / "pretrain.yaml", tmp_path / "artifacts")
    gate = _passing_gate(tmp_path / "gate.json")
    value = json.loads(gate.read_text(encoding="utf-8"))
    value["passed"] = False
    gate.write_text(json.dumps(value), encoding="utf-8")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("training setup should not be reached")

    monkeypatch.setattr(cli, "_prepare_training", forbidden)

    result = cli.main(["train", "pretrain", "--config", str(config), "--gate-record", str(gate)])

    assert result == 2
    assert not called
    assert "blocks formal pretraining" in capsys.readouterr().err


def test_pretrain_passes_resume_and_time_budget_to_training(tmp_path, monkeypatch, capsys) -> None:
    config = _write_pretrain_config(tmp_path / "pretrain.yaml", tmp_path / "artifacts")
    gate = _passing_gate(tmp_path / "gate.json", throughput=2.0)
    captured = {}

    def fake_prepare(configs, **kwargs):
        captured.update(kwargs)
        result = SimpleNamespace(
            stopped_reason="completed",
            best_checkpoint=tmp_path / "best.pt",
            last_checkpoint=tmp_path / "last.pt",
            metrics={"train/loss": 1.0},
        )
        return result, tmp_path / "metrics.jsonl"

    monkeypatch.setattr(cli, "_prepare_training", fake_prepare)

    result = cli.main(
        [
            "train",
            "pretrain",
            "--config",
            str(config),
            "--gate-record",
            str(gate),
            "--resume",
            str(tmp_path / "resume.pt"),
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert captured["optimizer_name"] == "muon"
    assert captured["resume"].endswith("resume.pt")
    assert captured["require_doctor"] is True
    assert captured["max_tokens"] == 32


def test_pretrain_rejects_repository_local_artifact_directories(tmp_path) -> None:
    config = _write_pretrain_config(
        tmp_path / "pretrain.yaml", cli._PROJECT_ROOT / "accidental-artifacts"
    )

    configs = cli.load_pretrain_config(config)
    with pytest.raises(ValueError, match="outside the Git repository"):
        cli._require_external_directory(configs.data.data_dir, label="data directory")
