from pathlib import Path

from scripts.audit_tracked_files import audit_paths


def test_audit_rejects_training_artifacts(tmp_path: Path) -> None:
    (tmp_path / "weights.safetensors").write_bytes(b"not real weights")
    violations = audit_paths(tmp_path, ["weights.safetensors", "data/train.jsonl"])
    assert any("extension" in item for item in violations)
    assert any("directory" in item for item in violations)


def test_audit_allows_source_and_small_fixture(tmp_path: Path) -> None:
    source = tmp_path / "src" / "package" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    assert audit_paths(tmp_path, ["src/package/model.py", "tests/fixtures/tiny.jsonl"]) == []


def test_audit_rejects_large_file(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * 11)
    assert audit_paths(tmp_path, ["large.txt"], max_bytes=10)
