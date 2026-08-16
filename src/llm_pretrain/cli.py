"""Command-line interface for the local 100M language-model training lab.

The CLI intentionally keeps network access opt-in and treats the A/B result as
an artefact that must be checked before a production pre-training run starts.
Heavy imports are project dependencies, but no model, CUDA tensor, or remote
dataset is created while parsing arguments or displaying help.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from .checkpoint import CheckpointManager, load_checkpoint, load_model_checkpoint, save_checkpoint
from .config import ConfigError, DataConfig, ModelConfig, OptimizerConfig, RunState, TrainConfig
from .data import (
    DEFAULT_SOURCES,
    DataCursor,
    Document,
    MemmapTokenDataset,
    SourceManifest,
    SourceSpec,
    allocate_token_quotas,
    download_wikimedia_dump,
    iter_huggingface_documents,
    iter_jsonl_documents,
    iter_wikimedia_documents,
    pack_mixed_token_shards,
    pack_token_shards,
    prepare_document_corpus,
    resolve_source_revisions,
)
from .doctor import DoctorError, require_training_ready, run_doctor
from .evaluation import evaluate
from .generation import GenerationConfig, generate_text, interactive_generate
from .model import CausalLM
from .optim import create_optimizers, create_scheduler_from_config
from .sft import (
    SFTConfig,
    SFTDataset,
    deterministic_coig_split,
    make_sft_collator,
    train_sft,
)
from .tokenization import (
    SentencePieceTokenizer,
    evaluate_tokenizer,
    train_sentencepiece,
    write_source_balanced_tokenizer_corpus,
    write_tokenizer_corpus,
)
from .training import (
    ABRunMetrics,
    LocalMetricLogger,
    evaluate_ab_gate,
    probe_micro_batch_sequences,
    read_nvidia_temperature_c,
    train,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRETRAIN_SECTIONS = frozenset({"model", "data", "optimizer", "train"})
_GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PretrainConfigs:
    """The four strictly validated sections of one pre-training YAML file."""

    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    source_path: Path


@dataclass(frozen=True, slots=True)
class _FreshValidationBatches:
    """Create a new finite memmap cursor for every validation pass."""

    manifest_path: Path
    batch_size: int
    sequence_length: int
    seed: int

    def __iter__(self) -> Iterator[Any]:
        return iter(
            MemmapTokenDataset(
                self.manifest_path,
                batch_size=self.batch_size,
                sequence_length=self.sequence_length,
                seed=self.seed,
                shuffle_shards=False,
                repeat=False,
            )
        )


def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(values, Mapping):
        raise ConfigError(f"configuration {source} must contain a YAML mapping")
    return dict(values)


def load_pretrain_config(path: str | Path) -> PretrainConfigs:
    """Load one nested pre-training YAML, rejecting unknown/missing sections."""

    source = Path(path)
    values = _read_yaml_mapping(source)
    unknown = sorted(set(values) - _PRETRAIN_SECTIONS)
    missing = sorted(_PRETRAIN_SECTIONS - set(values))
    if unknown:
        raise ConfigError(f"unknown pretrain section(s): {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"missing pretrain section(s): {', '.join(missing)}")
    configs = PretrainConfigs(
        model=ModelConfig.from_dict(values["model"]),
        data=DataConfig.from_dict(values["data"]),
        optimizer=OptimizerConfig.from_dict(values["optimizer"]),
        train=TrainConfig.from_dict(values["train"]),
        source_path=source.resolve(),
    )
    if configs.model.vocab_size != configs.data.tokenizer_vocab_size:
        raise ConfigError("model.vocab_size must equal data.tokenizer_vocab_size")
    if configs.model.max_seq_len != configs.data.sequence_length:
        raise ConfigError("model.max_seq_len must equal data.sequence_length")
    return configs


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_external_directory(path: str | Path, *, label: str) -> Path:
    resolved = _resolve_path(path)
    if _is_relative_to(resolved, _PROJECT_ROOT):
        raise ValueError(
            f"{label} must be outside the Git repository ({_PROJECT_ROOT}); "
            "set LLM_PRETRAIN_ARTIFACT_ROOT or pass an external path"
        )
    return resolved


def _with_artifact_root(configs: PretrainConfigs, explicit_root: str | None) -> PretrainConfigs:
    value = explicit_root or os.environ.get("LLM_PRETRAIN_ARTIFACT_ROOT")
    if not value:
        return configs
    root = _require_external_directory(value, label="artifact root")
    run_name = Path(configs.train.output_dir).name or "pretrain"
    return replace(
        configs,
        data=replace(configs.data, data_dir=str(root / "data")),
        train=replace(configs.train, output_dir=str(root / "runs" / run_name)),
    )


def _load_command_configs(args: argparse.Namespace) -> PretrainConfigs:
    return _with_artifact_root(load_pretrain_config(args.config), args.artifact_root)


def _json_dump(value: Any, *, output: str | Path | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def _parse_source_inputs(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--input must use SOURCE=PATH syntax")
        if name in parsed:
            raise ValueError(f"duplicate --input source: {name}")
        path = _resolve_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"local source file does not exist: {path}")
        parsed[name] = path
    return parsed


def _local_source_manifest(inputs: Mapping[str, Path], seed: int) -> SourceManifest:
    weight = 1.0 / len(inputs)
    sources = tuple(
        SourceSpec(
            name=name,
            repository=str(path),
            subset=None,
            revision="local",
            license="user-provided; verify before use",
            token_weight=weight,
            provider="local_jsonl",
        )
        for name, path in sorted(inputs.items())
    )
    return SourceManifest(sources, seed=seed)


def _legacy_data_settings(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    values = _read_yaml_mapping(path)
    if set(values) == _PRETRAIN_SECTIONS:
        return load_pretrain_config(path).data.to_dict()
    return values


def _artifact_root_from_args(args: argparse.Namespace, settings: Mapping[str, Any]) -> Path:
    explicit = args.artifact_root or os.environ.get("LLM_PRETRAIN_ARTIFACT_ROOT")
    raw = explicit or settings.get("artifact_root") or settings.get("data_dir")
    if not raw:
        raise ValueError(
            "an external artifact root is required; pass --artifact-root or set "
            "LLM_PRETRAIN_ARTIFACT_ROOT"
        )
    root = _require_external_directory(str(raw), label="artifact root")
    return root / "data" if explicit else root


def _command_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        args.data_dir,
        min_free_disk_gb=args.min_free_disk_gb,
        min_free_ram_gb=args.min_free_ram_gb,
        min_free_vram_gb=args.min_free_vram_gb,
        io_test_bytes=args.io_test_mib * 1024 * 1024,
    )
    sys.stdout.write(report.to_json() + "\n")
    if args.strict:
        report.require_ready()
    return 0 if report.ready else 1


def _limit_documents_by_utf8_bytes(
    records: Iterable[Document], max_utf8_bytes: int
) -> Iterator[Document]:
    """Stop a streaming source after a deterministic raw-text byte budget."""
    if max_utf8_bytes <= 0:
        raise ValueError("source UTF-8 byte budget must be positive")
    consumed = 0
    for document in records:
        consumed += len(document.text.encode("utf-8"))
        yield document
        if consumed >= max_utf8_bytes:
            return


def _network_records(
    sources: Sequence[SourceSpec],
    downloads: Path,
    *,
    source_utf8_bytes_per_token: float,
) -> dict[str, Iterable[Document]]:
    if source_utf8_bytes_per_token <= 0:
        raise ConfigError("source_utf8_bytes_per_token must be positive")
    records: dict[str, Iterable[Document]] = {}
    for source in sources:
        if source.provider == "huggingface":
            source_records: Iterable[Document] = iter_huggingface_documents(source)
        elif source.provider == "wikimedia_dump":
            dump = download_wikimedia_dump(source, downloads)
            source_records = iter_wikimedia_documents(source, dump)
        else:
            raise ValueError(f"unsupported remote source provider: {source.provider}")
        if source.requested_tokens is not None:
            byte_budget = math.ceil(source.requested_tokens * source_utf8_bytes_per_token)
            source_records = _limit_documents_by_utf8_bytes(source_records, byte_budget)
        records[source.name] = source_records
    return records


def _command_data_prepare(args: argparse.Namespace) -> int:
    settings = _legacy_data_settings(args.config)
    root = _artifact_root_from_args(args, settings)
    output = _require_external_directory(args.output_dir or root / "prepared", label="output")
    local_inputs = _parse_source_inputs(args.input)
    if not local_inputs and not args.allow_network:
        raise ValueError(
            "data preparation is offline by default; provide local --input SOURCE=PATH "
            "or explicitly pass --allow-network"
        )
    seed = int(settings.get("seed", 1337))
    if local_inputs:
        manifest = _local_source_manifest(local_inputs, seed)
        records = {name: _iter_local_jsonl(path) for name, path in local_inputs.items()}
    else:
        sources = resolve_source_revisions(DEFAULT_SOURCES)
        manifest = SourceManifest(sources, seed=seed)
        records = _network_records(
            sources,
            output / "downloads",
            source_utf8_bytes_per_token=float(settings.get("source_utf8_bytes_per_token", 8.0)),
        )
    stats = prepare_document_corpus(
        records,
        output,
        manifest=manifest,
        validation_fraction=args.validation_fraction,
    )
    _json_dump(asdict(stats))
    return 0


def _expand_inputs(explicit: Sequence[str], configured: Sequence[str]) -> list[Path]:
    patterns = [*configured, *explicit]
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        if not matches:
            # The checked-in tokenizer config predates the single-file corpus
            # writer and used train-*.jsonl/validation-*.jsonl patterns.
            fallback = pattern.replace("train-*", "train").replace("validation-*", "validation")
            if Path(fallback).is_file():
                matches = [fallback]
        paths.extend(_resolve_path(match) for match in matches)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError("no input documents matched; pass one or more --input paths")
    return unique


def _iter_texts(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            yield from (document.text for document in iter_jsonl_documents(path))
        else:
            with path.open("r", encoding="utf-8") as handle:
                yield from (line.strip() for line in handle if line.strip())


def _iter_local_jsonl(path: Path) -> Iterator[Document]:
    """Read text-only local fixtures; SOURCE= on the CLI supplies provenance."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                text = value["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"malformed local document at {path}:{line_number}") from exc
            if not isinstance(text, str):
                raise ValueError(f"local document text is not a string at {path}:{line_number}")
            identifier = value.get("id")
            yield Document(text, "local", None if identifier is None else str(identifier))


def _tokenizer_settings(path: str | Path | None) -> dict[str, Any]:
    return _read_yaml_mapping(path) if path is not None else {}


def _command_tokenizer_train(args: argparse.Namespace) -> int:
    settings = _tokenizer_settings(args.config)
    configured_inputs = settings.get("input_globs", ())
    if not isinstance(configured_inputs, Sequence) or isinstance(configured_inputs, str):
        raise ConfigError("tokenizer input_globs must be a sequence")
    artifact_root = args.artifact_root or os.environ.get("LLM_PRETRAIN_ARTIFACT_ROOT")
    if artifact_root and not args.input:
        configured_inputs = [
            str(_resolve_path(artifact_root) / "data" / "prepared" / "train.jsonl")
        ]
    inputs = _expand_inputs(args.input, [str(item) for item in configured_inputs])
    prefix_value = args.model_prefix
    if prefix_value is None and artifact_root:
        configured_name = Path(str(settings.get("output_prefix", "chinese-24k"))).name
        prefix_value = str(_resolve_path(artifact_root) / "data" / "tokenizer" / configured_name)
    if prefix_value is None:
        prefix_value = settings.get("output_prefix")
    if not prefix_value:
        raise ValueError("tokenizer model prefix is required via --model-prefix or config")
    prefix = _resolve_path(str(prefix_value))
    _require_external_directory(prefix.parent, label="tokenizer output directory")
    sample_bytes = int(settings.get("training_corpus_bytes", 256 * 1024 * 1024))
    if sample_bytes <= 0:
        raise ConfigError("tokenizer training_corpus_bytes must be positive")
    corpus_path = prefix.parent / f"{prefix.name}-train.txt"
    manifests = {path.parent / "sources.json" for path in inputs if path.suffix == ".jsonl"}
    all_jsonl = all(path.suffix == ".jsonl" for path in inputs)
    manifest_path = manifests.pop() if len(manifests) == 1 and all_jsonl else None
    if manifest_path is not None and manifest_path.is_file():
        manifest = SourceManifest.read(manifest_path)
        document_stream = (
            (document.source, document.text)
            for input_path in inputs
            for document in iter_jsonl_documents(input_path)
        )
        corpus_path, corpus_stats = write_source_balanced_tokenizer_corpus(
            document_stream,
            corpus_path,
            source_weights={source.name: source.token_weight for source in manifest.sources},
            max_utf8_bytes=sample_bytes,
        )
        corpus_details: dict[str, Any] = asdict(corpus_stats)
    else:
        corpus_path, document_count = write_tokenizer_corpus(
            _iter_texts(inputs), corpus_path, max_utf8_bytes=sample_bytes
        )
        corpus_details = {
            "documents": document_count,
            "utf8_bytes": corpus_path.stat().st_size,
        }
    model_path, vocab_path = train_sentencepiece(
        [corpus_path],
        prefix,
        vocab_size=int(
            args.vocab_size if args.vocab_size is not None else settings.get("vocab_size", 24_576)
        ),
        character_coverage=float(
            args.character_coverage
            if args.character_coverage is not None
            else settings.get("character_coverage", 0.99995)
        ),
        num_threads=args.num_threads,
    )
    _json_dump(
        {
            "model": str(model_path),
            "vocab": str(vocab_path),
            "training_corpus": str(corpus_path),
            "training_corpus_stats": corpus_details,
        }
    )
    return 0


def _command_tokenizer_eval(args: argparse.Namespace) -> int:
    settings = _tokenizer_settings(args.config)
    artifact_root = args.artifact_root or os.environ.get("LLM_PRETRAIN_ARTIFACT_ROOT")
    model_value = args.model
    if model_value is None and artifact_root:
        configured_name = Path(str(settings.get("output_prefix", "chinese-24k"))).name
        model_value = str(
            _resolve_path(artifact_root) / "data" / "tokenizer" / f"{configured_name}.model"
        )
    if model_value is None:
        model_value = settings.get("model")
    if model_value is None and settings.get("output_prefix"):
        model_value = str(settings["output_prefix"]) + ".model"
    if not model_value:
        raise ValueError("tokenizer model is required via --model or config output_prefix")
    configured_inputs = settings.get("validation_globs")
    if configured_inputs is None:
        training_inputs = settings.get("input_globs", ())
        configured_inputs = [
            str(item).replace("train-*", "validation-*").replace("train.jsonl", "validation.jsonl")
            for item in training_inputs
        ]
    if artifact_root and not args.input:
        configured_inputs = [
            str(_resolve_path(artifact_root) / "data" / "prepared" / "validation.jsonl")
        ]
    if not isinstance(configured_inputs, Sequence) or isinstance(configured_inputs, str):
        raise ConfigError("tokenizer validation_globs must be a sequence")
    inputs = _expand_inputs(args.input, [str(item) for item in configured_inputs])
    metrics = evaluate_tokenizer(SentencePieceTokenizer.from_file(model_value), _iter_texts(inputs))
    _json_dump(metrics.to_dict(), output=args.output)
    return 0


def _command_data_tokenize(args: argparse.Namespace) -> int:
    settings = _legacy_data_settings(args.config)
    root = _artifact_root_from_args(args, settings)
    prepared = _resolve_path(args.prepared_dir or root / "prepared")
    output = _require_external_directory(args.output_dir or root / "shards", label="output")
    tokenizer_path = _resolve_path(args.tokenizer or root / "tokenizer" / "chinese-24k.model")
    tokenizer = SentencePieceTokenizer.from_file(tokenizer_path)
    sequence_length = int(args.sequence_length or settings.get("sequence_length", 1024))
    shard_capacity = int(args.shard_size_tokens or settings.get("shard_size_tokens", 100_000_000))
    source_manifest_path = prepared / "sources.json"
    source_manifest = (
        SourceManifest.read(source_manifest_path) if source_manifest_path.is_file() else None
    )
    default_train_tokens: int | None = None
    if source_manifest is not None:
        requested = [source.requested_tokens for source in source_manifest.sources]
        if requested and all(value is not None for value in requested):
            default_train_tokens = sum(int(value) for value in requested if value is not None)
    train_tokens = args.train_tokens or default_train_tokens
    validation_tokens = args.validation_tokens or int(settings.get("validation_tokens", 10_000_000))
    manifests = {}
    for split in ("train", "validation"):
        source = prepared / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"prepared split does not exist: {source}")
        target_tokens = train_tokens if split == "train" else validation_tokens
        if source_manifest is not None and target_tokens is not None:
            quotas = allocate_token_quotas(target_tokens, source_manifest.sources)
            manifest = pack_mixed_token_shards(
                iter_jsonl_documents(source),
                tokenizer,
                output,
                split=split,
                source_token_quotas=quotas,
                sequence_length=sequence_length,
                shard_token_capacity=shard_capacity,
            )
        else:
            manifest = pack_token_shards(
                iter_jsonl_documents(source),
                tokenizer,
                output,
                split=split,
                sequence_length=sequence_length,
                shard_token_capacity=shard_capacity,
            )
        manifests[split] = manifest.to_dict()
    _json_dump(manifests)
    return 0


def _model_and_tokenizer(
    configs: PretrainConfigs,
    checkpoint: str | Path,
    tokenizer_path: str | Path | None,
) -> tuple[CausalLM, SentencePieceTokenizer]:
    model = CausalLM(configs.model)
    load_model_checkpoint(checkpoint, model, map_location="cpu")
    data_root = _resolve_path(configs.data.data_dir)
    tokenizer = SentencePieceTokenizer.from_file(
        tokenizer_path or data_root / "tokenizer" / "chinese-24k.model"
    )
    return model, tokenizer


def _prepare_training(
    configs: PretrainConfigs,
    *,
    output_dir: Path,
    optimizer_name: str,
    max_tokens: int | None,
    resume: str | None,
    require_doctor: bool,
) -> tuple[Any, Path]:
    data_dir = _require_external_directory(configs.data.data_dir, label="data directory")
    output_dir = _require_external_directory(output_dir, label="training output directory")
    if require_doctor:
        require_training_ready(data_dir=data_dir, min_free_vram_gb=configs.train.memory_margin_gb)
    torch.manual_seed(configs.train.seed)
    model = CausalLM(configs.model)
    train_config = replace(
        configs.train,
        output_dir=str(output_dir),
        max_tokens=max_tokens or configs.train.max_tokens,
    )
    micro_batch = train_config.micro_batch_sequences
    if micro_batch == 0:
        micro_batch = probe_micro_batch_sequences(
            model,
            sequence_length=configs.data.sequence_length,
            candidates=train_config.micro_batch_candidates,
            device=train_config.device,
            memory_margin_gb=train_config.memory_margin_gb,
            use_bf16=train_config.dtype == "bfloat16",
        )
        train_config = replace(train_config, micro_batch_sequences=micro_batch)

    shard_dir = data_dir / "shards"
    train_data = MemmapTokenDataset(
        shard_dir / "train-shards.json",
        batch_size=micro_batch,
        sequence_length=configs.data.sequence_length,
        seed=configs.data.seed,
    )
    validation_data = _FreshValidationBatches(
        shard_dir / "validation-shards.json",
        batch_size=micro_batch,
        sequence_length=configs.data.sequence_length,
        seed=configs.data.seed,
    )
    model.to(train_config.device)
    optimizer_config = replace(configs.optimizer, optimizer=optimizer_name)
    optimizer = create_optimizers(model, optimizer_config)
    total_updates = math.ceil(train_config.max_tokens / train_config.global_batch_tokens)
    scheduler = create_scheduler_from_config(optimizer, total_updates, optimizer_config)
    state = RunState(optimizer=optimizer_name)
    if resume:
        load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_state=state,
            map_location="cpu",
        )
        train_data.load_state_dict(
            DataCursor(
                epoch=state.epoch,
                shard_position=state.data_shard_index,
                token_offset=state.data_offset,
            ).state_dict()
        )
    manager = CheckpointManager(
        output_dir / "checkpoints", keep_last=train_config.keep_last_checkpoints
    )
    log_path = output_dir / "metrics.jsonl"
    runtime_config = {
        **train_config.to_dict(),
        "sequence_length": configs.data.sequence_length,
    }
    with LocalMetricLogger(log_path, output_dir / "tensorboard") as logger:
        result = train(
            model,
            train_data,
            optimizer,
            runtime_config,
            scheduler=scheduler,
            validation_loader=validation_data,
            checkpoint_manager=manager,
            metric_logger=logger,
            run_state=state,
            device=train_config.device,
            temperature_reader=(
                read_nvidia_temperature_c if train_config.device == "cuda" else None
            ),
        )
    return result, log_path


def _read_ab_metrics(log_path: Path, stopped_reason: str) -> ABRunMetrics:
    validation_losses: list[float] = []
    throughputs: list[float] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if "validation/loss" in event:
                validation_losses.append(float(event["validation/loss"]))
            if "train/tokens_per_second" in event:
                throughputs.append(float(event["train/tokens_per_second"]))
    return ABRunMetrics(
        validation_losses=tuple(validation_losses),
        tokens_per_second=sum(throughputs) / len(throughputs) if throughputs else 0.0,
        finite=stopped_reason == "completed"
        and all(math.isfinite(value) for value in validation_losses + throughputs),
    )


def _command_train_ab(args: argparse.Namespace) -> int:
    configs = _load_command_configs(args)
    base_output = _require_external_directory(configs.train.output_dir, label="output") / "ab"
    gate_path = Path(args.gate_record or base_output / "gate.json")
    if gate_path.exists():
        raise ValueError(
            f"A/B gate record already exists: {gate_path}; keep the audit record and "
            "use a new output directory for another experiment"
        )
    data_dir = _require_external_directory(configs.data.data_dir, label="data directory")
    require_training_ready(
        data_dir=data_dir,
        min_free_vram_gb=configs.train.memory_margin_gb,
    )
    target_tokens = args.max_tokens or min(configs.train.max_tokens, 100_000_000)
    total_updates = math.ceil(target_tokens / configs.train.global_batch_tokens)
    # Ensure a normal 100M run yields at least three validation points for the gate.
    ab_train = replace(
        configs.train,
        eval_interval_steps=min(configs.train.eval_interval_steps, max(1, total_updates // 3)),
    )
    configs = replace(configs, train=ab_train)
    run_metrics: dict[str, ABRunMetrics] = {}
    for optimizer_name in ("adamw", "muon"):
        result, log_path = _prepare_training(
            configs,
            output_dir=base_output / optimizer_name,
            optimizer_name=optimizer_name,
            max_tokens=target_tokens,
            resume=None,
            require_doctor=False,
        )
        run_metrics[optimizer_name] = _read_ab_metrics(log_path, result.stopped_reason)
    decision = evaluate_ab_gate(run_metrics["muon"], run_metrics["adamw"])
    initial_runs = dict(run_metrics)
    calibrations: list[dict[str, Any]] = []
    if not decision.passed:
        calibration_tokens = min(50_000_000, max(1, target_tokens // 2))
        calibration_updates = math.ceil(calibration_tokens / configs.train.global_batch_tokens)
        calibration_train = replace(
            configs.train,
            eval_interval_steps=min(
                configs.train.eval_interval_steps,
                max(1, calibration_updates // 3),
            ),
        )
        candidates: list[tuple[float, ABRunMetrics]] = []
        for learning_rate in (0.01, 0.04):
            calibration_configs = replace(
                configs,
                train=calibration_train,
                optimizer=replace(configs.optimizer, muon_lr=learning_rate),
            )
            result, log_path = _prepare_training(
                calibration_configs,
                output_dir=base_output / f"calibration-muon-{learning_rate:g}",
                optimizer_name="muon",
                max_tokens=calibration_tokens,
                resume=None,
                require_doctor=False,
            )
            metrics = _read_ab_metrics(log_path, result.stopped_reason)
            calibrations.append(
                {
                    "muon_lr": learning_rate,
                    "target_tokens": calibration_tokens,
                    "metrics": asdict(metrics),
                }
            )
            if metrics.finite and metrics.validation_losses:
                candidates.append((learning_rate, metrics))
        if candidates:
            selected_lr, _ = min(
                candidates,
                key=lambda item: (
                    sum(item[1].validation_losses[-3:]) / len(item[1].validation_losses[-3:])
                ),
            )
            retry_configs = replace(
                configs,
                optimizer=replace(configs.optimizer, muon_lr=selected_lr),
            )
            result, log_path = _prepare_training(
                retry_configs,
                output_dir=base_output / f"muon-retry-{selected_lr:g}",
                optimizer_name="muon",
                max_tokens=target_tokens,
                resume=None,
                require_doctor=False,
            )
            run_metrics["muon"] = _read_ab_metrics(log_path, result.stopped_reason)
            decision = evaluate_ab_gate(run_metrics["muon"], run_metrics["adamw"])
    record = {
        "schema_version": _GATE_SCHEMA_VERSION,
        "passed": decision.passed,
        "decision": asdict(decision),
        "target_tokens_per_run": target_tokens,
        "runs": {name: asdict(metrics) for name, metrics in run_metrics.items()},
        "initial_runs": {name: asdict(metrics) for name, metrics in initial_runs.items()},
        "calibrations": calibrations,
    }
    _json_dump(record, output=gate_path)
    _json_dump({**record, "gate_record": str(gate_path.resolve())})
    return 0 if decision.passed else 1


def validate_gate_record(path: str | Path) -> dict[str, Any]:
    """Load a locally produced A/B record and require an explicit passing decision."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read A/B gate record {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"A/B gate record is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("A/B gate record must be a JSON object")
    if value.get("schema_version") != _GATE_SCHEMA_VERSION:
        raise ValueError(f"A/B gate record schema_version must be {_GATE_SCHEMA_VERSION}")
    decision = value.get("decision")
    runs = value.get("runs")
    if not isinstance(decision, dict) or not isinstance(runs, dict):
        raise ValueError("A/B gate record is missing decision or runs")
    if set(runs) != {"muon", "adamw"}:
        raise ValueError("A/B gate record must contain exactly muon and adamw runs")
    target_tokens = value.get("target_tokens_per_run")
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool):
        raise ValueError("A/B gate target_tokens_per_run must be an integer")
    if target_tokens < 100_000_000:
        raise ValueError("A/B gate requires at least 100,000,000 tokens per run")
    if value.get("passed") is not True or decision.get("passed") is not True:
        reasons = decision.get("reasons") or ["gate did not pass"]
        raise ValueError(f"A/B gate blocks formal pretraining: {reasons}")
    parsed_runs: dict[str, ABRunMetrics] = {}
    for name in ("muon", "adamw"):
        metrics = runs[name]
        if not isinstance(metrics, dict) or metrics.get("finite") is not True:
            raise ValueError(f"A/B gate {name} run is absent or non-finite")
        losses = metrics.get("validation_losses")
        throughput = metrics.get("tokens_per_second")
        if not isinstance(losses, list) or len(losses) < 3:
            raise ValueError(f"A/B gate {name} run needs at least three validation losses")
        if not isinstance(throughput, (int, float)) or isinstance(throughput, bool):
            raise ValueError(f"A/B gate {name} throughput is not numeric")
        try:
            numbers = [float(item) for item in losses] + [float(throughput)]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"A/B gate {name} metrics are not numeric") from exc
        if not all(math.isfinite(item) for item in numbers) or numbers[-1] <= 0:
            raise ValueError(f"A/B gate {name} metrics must be finite and throughput positive")
        parsed_runs[name] = ABRunMetrics(tuple(numbers[:-1]), numbers[-1], finite=True)
    recalculated = evaluate_ab_gate(parsed_runs["muon"], parsed_runs["adamw"])
    if not recalculated.passed:
        raise ValueError(
            f"A/B gate metrics do not pass formal criteria: {list(recalculated.reasons)}"
        )
    return value


def calculate_formal_token_budget(
    measured_tokens_per_second: float,
    *,
    time_budget_hours: float,
    maximum_tokens: int,
    global_batch_tokens: int,
    requested_max_tokens: int | None = None,
) -> int:
    """Apply the 90% wall-clock budget and align it to complete optimizer steps."""

    if not math.isfinite(measured_tokens_per_second) or measured_tokens_per_second <= 0:
        raise ValueError("measured tokens per second must be finite and positive")
    if time_budget_hours <= 0 or maximum_tokens <= 0 or global_batch_tokens <= 0:
        raise ValueError("formal training budget inputs must be positive")
    wall_clock_tokens = int(measured_tokens_per_second * time_budget_hours * 3600 * 0.9)
    upper_bound = min(maximum_tokens, wall_clock_tokens)
    if requested_max_tokens is not None:
        if requested_max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        upper_bound = min(upper_bound, requested_max_tokens)
    aligned = upper_bound // global_batch_tokens * global_batch_tokens
    if aligned <= 0:
        raise ValueError("measured throughput does not cover one global token batch")
    return aligned


def _command_train_pretrain(args: argparse.Namespace) -> int:
    configs = _load_command_configs(args)
    if configs.optimizer.optimizer != "muon":
        raise ConfigError("formal pretraining requires a config with optimizer: muon")
    gate = validate_gate_record(args.gate_record)
    measured_tps = float(gate["runs"]["muon"]["tokens_per_second"])
    target_tokens = calculate_formal_token_budget(
        measured_tps,
        time_budget_hours=configs.train.time_budget_hours,
        maximum_tokens=configs.train.max_tokens,
        global_batch_tokens=configs.train.global_batch_tokens,
        requested_max_tokens=args.max_tokens,
    )
    result, _ = _prepare_training(
        configs,
        output_dir=_resolve_path(configs.train.output_dir),
        optimizer_name="muon",
        max_tokens=target_tokens,
        resume=args.resume,
        require_doctor=True,
    )
    _json_dump(
        {
            "stopped_reason": result.stopped_reason,
            "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
            "last_checkpoint": str(result.last_checkpoint) if result.last_checkpoint else None,
            "target_tokens": target_tokens,
            "metrics": result.metrics,
        }
    )
    return 0 if result.stopped_reason in {"completed", "time_budget"} else 1


def _checkpoint_default(configs: PretrainConfigs) -> Path:
    return _resolve_path(configs.train.output_dir) / "checkpoints" / "best.pt"


def _command_evaluate(args: argparse.Namespace) -> int:
    configs = _load_command_configs(args)
    checkpoint = args.checkpoint or _checkpoint_default(configs)
    model, tokenizer = _model_and_tokenizer(configs, checkpoint, args.tokenizer)
    model.to(configs.train.device)
    batch_size = args.batch_size
    validation = MemmapTokenDataset(
        _resolve_path(configs.data.data_dir) / "shards" / "validation-shards.json",
        batch_size=batch_size,
        sequence_length=configs.data.sequence_length,
        shuffle_shards=False,
        repeat=False,
    )
    result = evaluate(
        model,
        tokenizer,
        validation,
        device=configs.train.device,
        max_batches=args.max_batches,
        use_bf16=configs.train.dtype == "bfloat16",
    )
    _json_dump(result.to_dict(), output=args.output)
    return 0


def _load_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"SFT record at {path}:{line_number} is not an object")
            yield value


def _command_train_sft(args: argparse.Namespace) -> int:
    settings = _read_yaml_mapping(args.config)
    base_config = args.base_config or settings.get("base_config")
    if not base_config:
        raise ConfigError("SFT config requires base_config")
    configs = _with_artifact_root(load_pretrain_config(base_config), args.artifact_root)
    checkpoint = args.checkpoint or _checkpoint_default(configs)
    model, tokenizer = _model_and_tokenizer(configs, checkpoint, args.tokenizer)
    if args.records:
        records: Iterable[Mapping[str, Any]] = _load_jsonl_records(_resolve_path(args.records))
    elif args.allow_network:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("datasets is required to download COIG") from exc
        dataset = load_dataset(
            str(settings.get("dataset_id", "BAAI/COIG")),
            split="train",
            revision=str(settings.get("revision", "main")),
            streaming=True,
        )
        records = dataset
    else:
        raise ValueError("SFT is offline by default; pass --records LOCAL.jsonl or --allow-network")
    sft_config = SFTConfig(
        learning_rate=float(settings.get("learning_rate", 5e-5)),
        epochs=int(settings.get("epochs", 2)),
        max_length=int(settings.get("max_seq_len", 1024)),
        train_examples=int(
            args.train_examples
            if args.train_examples is not None
            else settings.get("train_examples", 50_000)
        ),
        validation_examples=int(
            args.validation_examples
            if args.validation_examples is not None
            else settings.get("validation_examples", 1_000)
        ),
        seed=int(settings.get("seed", 42)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )
    torch.manual_seed(sft_config.seed)
    split = deterministic_coig_split(
        records,
        train_size=sft_config.train_examples,
        validation_size=sft_config.validation_examples,
        seed=sft_config.seed,
    )
    from torch.utils.data import DataLoader

    collator = make_sft_collator(tokenizer.pad_id)
    train_loader = DataLoader(
        SFTDataset(split.train, tokenizer, max_length=sft_config.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        SFTDataset(split.validation, tokenizer, max_length=sft_config.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    result = train_sft(
        model,
        train_loader,
        validation_batches=validation_loader,
        config=sft_config,
        device=configs.train.device,
    )
    artifact_root = args.artifact_root or os.environ.get("LLM_PRETRAIN_ARTIFACT_ROOT")
    raw_output = args.output_dir
    if raw_output is None and artifact_root:
        raw_output = _resolve_path(artifact_root) / "runs" / "sft-coig"
    if raw_output is None:
        raw_output = settings.get("output_dir")
    if not raw_output:
        raise ConfigError("SFT output_dir is required")
    output_dir = _require_external_directory(raw_output, label="SFT output directory")
    checkpoint_path = save_checkpoint(
        output_dir / "best.pt", model, metrics={"sft": result.to_dict()}
    )
    _json_dump({**result.to_dict(), "checkpoint": str(checkpoint_path)})
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    configs = _load_command_configs(args)
    checkpoint = args.checkpoint or _checkpoint_default(configs)
    model, tokenizer = _model_and_tokenizer(configs, checkpoint, args.tokenizer)
    model.to(configs.train.device)
    settings = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    if args.interactive:
        interactive_generate(model, tokenizer, settings)
    else:
        if args.prompt is None:
            raise ValueError("--prompt is required unless --interactive is used")
        result = generate_text(model, tokenizer, args.prompt, settings)
        sys.stdout.write(result.completion + "\n")
    return 0


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-pretrain",
        description="Reproducible single-GPU 100M Chinese LLM training lab",
    )
    parser.add_argument(
        "--artifact-root",
        help="external data/run root (or set LLM_PRETRAIN_ARTIFACT_ROOT)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check WSL, CUDA, Muon and storage")
    doctor.add_argument("--data-dir")
    doctor.add_argument("--strict", action="store_true", help="fail when any check fails")
    doctor.add_argument("--min-free-disk-gb", type=float, default=50.0)
    doctor.add_argument("--min-free-ram-gb", type=float, default=4.0)
    doctor.add_argument("--min-free-vram-gb", type=float, default=0.7)
    doctor.add_argument("--io-test-mib", type=int, default=4)
    doctor.set_defaults(handler=_command_doctor)

    data = commands.add_parser("data", help="prepare documents and packed token shards")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare", help="normalize, dedupe and split documents")
    prepare.add_argument("--config")
    prepare.add_argument("--input", action="append", default=[], metavar="SOURCE=PATH")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--validation-fraction", type=float, default=0.005)
    prepare.add_argument("--allow-network", action="store_true")
    prepare.set_defaults(handler=_command_data_prepare)
    tokenize = data_commands.add_parser("tokenize", help="pack prepared splits as uint32 shards")
    tokenize.add_argument("--config")
    tokenize.add_argument("--prepared-dir")
    tokenize.add_argument("--output-dir")
    tokenize.add_argument("--tokenizer")
    tokenize.add_argument("--sequence-length", type=int)
    tokenize.add_argument("--shard-size-tokens", type=int)
    tokenize.add_argument("--train-tokens", type=int)
    tokenize.add_argument("--validation-tokens", type=int)
    tokenize.set_defaults(handler=_command_data_tokenize)

    tokenizer = commands.add_parser("tokenizer", help="train or evaluate SentencePiece")
    tokenizer_commands = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    tokenizer_train = tokenizer_commands.add_parser("train", help="train the fixed-id BPE")
    tokenizer_train.add_argument("--config")
    tokenizer_train.add_argument("--input", action="append", default=[])
    tokenizer_train.add_argument("--model-prefix")
    tokenizer_train.add_argument("--vocab-size", type=int)
    tokenizer_train.add_argument("--character-coverage", type=float)
    tokenizer_train.add_argument("--num-threads", type=int)
    tokenizer_train.set_defaults(handler=_command_tokenizer_train)
    tokenizer_eval = tokenizer_commands.add_parser("eval", help="measure held-out compression")
    tokenizer_eval.add_argument("--config")
    tokenizer_eval.add_argument("--input", action="append", default=[])
    tokenizer_eval.add_argument("--model")
    tokenizer_eval.add_argument("--output")
    tokenizer_eval.set_defaults(handler=_command_tokenizer_eval)

    training = commands.add_parser("train", help="run A/B, pre-training, or SFT")
    training_commands = training.add_subparsers(dest="train_command", required=True)
    ab = training_commands.add_parser("ab", help="compare AdamW and Muon and write a gate")
    _add_config_argument(ab)
    ab.add_argument("--max-tokens", type=int)
    ab.add_argument("--gate-record")
    ab.set_defaults(handler=_command_train_ab)
    pretrain = training_commands.add_parser(
        "pretrain",
        help="formal Muon pre-training (requires passing A/B gate)",
        description="Formal Muon pre-training; requires passing A/B gate and strict doctor.",
    )
    _add_config_argument(pretrain)
    pretrain.add_argument("--gate-record", required=True)
    pretrain.add_argument("--resume")
    pretrain.add_argument("--max-tokens", type=int)
    pretrain.set_defaults(handler=_command_train_pretrain)
    sft = training_commands.add_parser("sft", help="full-parameter assistant-only COIG SFT")
    _add_config_argument(sft)
    sft.add_argument("--base-config")
    sft.add_argument("--checkpoint")
    sft.add_argument("--tokenizer")
    sft.add_argument("--records")
    sft.add_argument("--allow-network", action="store_true")
    sft.add_argument("--train-examples", type=int)
    sft.add_argument("--validation-examples", type=int)
    sft.add_argument("--batch-size", type=int, default=1)
    sft.add_argument("--output-dir")
    sft.set_defaults(handler=_command_train_sft)

    evaluation = commands.add_parser("evaluate", help="held-out loss and fixed prompts")
    _add_config_argument(evaluation)
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--tokenizer")
    evaluation.add_argument("--batch-size", type=int, default=1)
    evaluation.add_argument("--max-batches", type=int)
    evaluation.add_argument("--output")
    evaluation.set_defaults(handler=_command_evaluate)

    generation = commands.add_parser("generate", help="generate once or interactively")
    _add_config_argument(generation)
    generation.add_argument("--checkpoint")
    generation.add_argument("--tokenizer")
    generation.add_argument("--prompt")
    generation.add_argument("--interactive", action="store_true")
    generation.add_argument("--max-new-tokens", type=int, default=128)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=50)
    generation.add_argument("--seed", type=int, default=42)
    generation.set_defaults(handler=_command_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, DoctorError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
