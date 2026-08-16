"""Fail when Git tracks local training artifacts, secrets, or oversized files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

FORBIDDEN_DIRS = {
    "checkpoints",
    "data",
    "datasets",
    "logs",
    "runs",
    "tokenized",
    "tokenizers",
    "wandb",
}
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".idx",
    ".model",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".tfrecord",
}
SECRET_NAMES = {".env", "credentials.json", "secrets.json", "token.txt"}
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def audit_paths(repo_root: Path, paths: list[str], max_bytes: int = DEFAULT_MAX_BYTES) -> list[str]:
    """Return human-readable violations for tracked relative paths."""

    violations: list[str] = []
    for raw in paths:
        normalized = raw.replace("\\", "/").lstrip("./")
        path = PurePosixPath(normalized)
        parts = set(path.parts)
        local = repo_root / Path(*path.parts)

        if parts & FORBIDDEN_DIRS:
            violations.append(f"forbidden artifact directory: {normalized}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact extension: {normalized}")
        if path.name.lower() in SECRET_NAMES or path.name.lower().endswith(".secret"):
            violations.append(f"possible secret file: {normalized}")
        if local.is_file() and local.stat().st_size > max_bytes:
            violations.append(
                f"oversized tracked file ({local.stat().st_size} bytes > {max_bytes}): {normalized}"
            )
    return violations


def tracked_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    try:
        violations = audit_paths(root, tracked_paths(root), args.max_bytes)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"tracked-file audit failed to inspect Git: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("Tracked-file audit failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("Tracked-file audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
