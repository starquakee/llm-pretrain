"""Deterministic document preparation and packed-token shard readers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from llm_pretrain.tokenization import TokenizerProtocol

DEFAULT_SEED = 19_907_023
DEFAULT_SEQUENCE_LENGTH = 1024
UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class SourceFile:
    """A materialized upstream file captured in a reproducibility manifest."""

    path: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class SourceSpec:
    """An immutable description of one pretraining source."""

    name: str
    repository: str
    subset: str | None
    revision: str
    license: str
    token_weight: float
    requested_tokens: int | None = None
    text_field: str = "text"
    split: str = "train"
    files: tuple[SourceFile, ...] = ()
    provider: str = "huggingface"


# The symbolic Hugging Face revisions are resolved to immutable commit hashes by
# ``resolve_source_revisions`` before a production manifest is written.  The
# Wikimedia dump is date-versioned and therefore already immutable.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="ultra_fineweb_zh",
        repository="openbmb/Ultra-FineWeb",
        subset="default",
        revision="main",
        license="Apache-2.0",
        token_weight=0.80,
        requested_tokens=1_600_000_000,
        text_field="content",
        split="zh",
    ),
    SourceSpec(
        name="wikimedia_zh",
        repository="https://dumps.wikimedia.org/zhwiki/",
        subset="pages-articles-multistream",
        revision="2026-08-01",
        license="CC BY-SA 4.0 and GFDL; contributor terms apply",
        token_weight=0.10,
        requested_tokens=200_000_000,
        split="pages",
        files=(SourceFile(path="zhwiki-20260801-pages-articles-multistream.xml.bz2"),),
        provider="wikimedia_dump",
    ),
    SourceSpec(
        name="fineweb_en",
        repository="HuggingFaceFW/fineweb",
        subset="sample-10BT",
        revision="main",
        license="ODC-By v1.0; Common Crawl terms and source rights apply",
        token_weight=0.10,
        requested_tokens=200_000_000,
    ),
)


@dataclass(frozen=True)
class SourceManifest:
    """Pinned inputs and mixing policy for one prepared corpus."""

    sources: tuple[SourceSpec, ...]
    seed: int = DEFAULT_SEED
    normalization: str = "NFKC+newline/whitespace canonicalization v1"
    version: int = 1

    def validate(self, *, require_pinned_revisions: bool = True) -> None:
        if not self.sources:
            raise ValueError("Source manifest cannot be empty")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")
        total_weight = sum(source.token_weight for source in self.sources)
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Source token weights must sum to 1.0, got {total_weight}")
        if any(source.token_weight <= 0.0 for source in self.sources):
            raise ValueError("Source token weights must be positive")
        requested = [source.requested_tokens for source in self.sources]
        if any(value is not None for value in requested):
            if any(value is None or value <= 0 for value in requested):
                raise ValueError("Requested token counts must be positive and set for every source")
            total_requested = sum(value for value in requested if value is not None)
            for source in self.sources:
                assert source.requested_tokens is not None
                expected_weight = source.requested_tokens / total_requested
                if not math.isclose(
                    expected_weight, source.token_weight, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Requested tokens for {source.name} do not match its token weight"
                    )
        if require_pinned_revisions:
            symbolic = [
                source.name for source in self.sources if source.revision in {"main", "master"}
            ]
            if symbolic:
                raise ValueError(
                    f"Resolve symbolic source revisions before preparing data: {symbolic}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path, *, require_pinned_revisions: bool = True) -> Path:
        self.validate(require_pinned_revisions=require_pinned_revisions)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(output, self.to_dict())
        return output

    @classmethod
    def read(cls, path: str | Path) -> SourceManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = []
        for item in payload["sources"]:
            item = dict(item)
            item["files"] = tuple(SourceFile(**file_item) for file_item in item.get("files", ()))
            sources.append(SourceSpec(**item))
        return cls(
            sources=tuple(sources),
            seed=int(payload["seed"]),
            normalization=str(payload["normalization"]),
            version=int(payload["version"]),
        )


def resolve_source_revisions(
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
    *,
    resolver: Callable[[str, str], str] | None = None,
) -> tuple[SourceSpec, ...]:
    """Resolve symbolic Hugging Face revisions to immutable commit hashes.

    Tests can inject ``resolver(repository, revision) -> sha``.  Production use
    lazily imports ``huggingface_hub`` and therefore performs no network access
    until this function is explicitly called.
    """
    revision_resolver = resolver
    if revision_resolver is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("huggingface_hub is required to resolve dataset revisions") from exc
        api = HfApi()

        def resolve_huggingface_revision(repository: str, revision: str) -> str:
            return str(api.dataset_info(repository, revision=revision).sha)

        revision_resolver = resolve_huggingface_revision

    resolved: list[SourceSpec] = []
    for source in sources:
        if source.provider == "huggingface" and source.revision in {"main", "master"}:
            sha = revision_resolver(source.repository, source.revision)
            if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                raise ValueError(
                    f"Resolver returned a non-commit revision for {source.name}: {sha}"
                )
            source_payload = asdict(source)
            source_payload["revision"] = sha.lower()
            source_payload["files"] = tuple(source.files)
            resolved.append(SourceSpec(**source_payload))
        else:
            resolved.append(source)
    return tuple(resolved)


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    document_id: str | None = None


_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_document(text: str) -> str:
    """Canonicalize text before hashing, splitting and exact deduplication."""
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")
    )
    return _EXCESS_NEWLINES.sub("\n\n", normalized).strip()


def document_sha256(text: str, *, already_normalized: bool = False) -> str:
    normalized = text if already_normalized else normalize_document(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_from_hash(
    sha256: str,
    *,
    validation_fraction: float,
    seed: int = DEFAULT_SEED,
) -> str:
    """Assign an exact document hash to train/validation deterministically."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    try:
        bytes.fromhex(sha256)
    except ValueError as exc:
        raise ValueError("sha256 must be hexadecimal") from exc
    if len(sha256) != 64:
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    mixed = hashlib.sha256(f"{seed}:{sha256.lower()}".encode()).digest()
    unit_interval = int.from_bytes(mixed[:8], "big") / 2**64
    return "validation" if unit_interval < validation_fraction else "train"


@dataclass(frozen=True)
class PreparationStats:
    input_documents: int
    train_documents: int
    validation_documents: int
    duplicate_documents: int
    empty_documents: int
    by_source: dict[str, int] = field(default_factory=dict)


def _coerce_document(record: Document | str | Mapping[str, Any], source: SourceSpec) -> Document:
    if isinstance(record, Document):
        return Document(record.text, source.name, record.document_id)
    if isinstance(record, str):
        return Document(record, source.name)
    text = record.get(source.text_field)
    if not isinstance(text, str):
        raise ValueError(f"Record from {source.name} has no string `{source.text_field}` field")
    document_id = record.get("id")
    return Document(text, source.name, None if document_id is None else str(document_id))


def prepare_document_corpus(
    records_by_source: Mapping[str, Iterable[Document | str | Mapping[str, Any]]],
    output_dir: str | Path,
    *,
    manifest: SourceManifest,
    validation_fraction: float,
    require_pinned_revisions: bool = True,
) -> PreparationStats:
    """Normalize, globally exact-dedupe, hash-split and write JSONL partitions."""
    manifest.validate(require_pinned_revisions=require_pinned_revisions)
    known_sources = {source.name for source in manifest.sources}
    unexpected = set(records_by_source) - known_sources
    if unexpected:
        raise ValueError(f"Records provided for unknown sources: {sorted(unexpected)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    seen: set[str] = set()
    counts = {source.name: 0 for source in manifest.sources}
    input_documents = duplicates = empty = train_count = validation_count = 0

    train_tmp = train_path.with_suffix(".jsonl.tmp")
    validation_tmp = validation_path.with_suffix(".jsonl.tmp")
    try:
        with (
            train_tmp.open("w", encoding="utf-8", newline="\n") as train_file,
            validation_tmp.open("w", encoding="utf-8", newline="\n") as validation_file,
        ):
            for source in manifest.sources:
                for raw_record in records_by_source.get(source.name, ()):
                    input_documents += 1
                    document = _coerce_document(raw_record, source)
                    text = normalize_document(document.text)
                    if not text:
                        empty += 1
                        continue
                    digest = document_sha256(text, already_normalized=True)
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    split = split_from_hash(
                        digest,
                        validation_fraction=validation_fraction,
                        seed=manifest.seed,
                    )
                    payload = {
                        "id": document.document_id or digest,
                        "sha256": digest,
                        "source": source.name,
                        "text": text,
                    }
                    destination = validation_file if split == "validation" else train_file
                    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    destination.write(serialized + "\n")
                    counts[source.name] += 1
                    if split == "validation":
                        validation_count += 1
                    else:
                        train_count += 1
        os.replace(train_tmp, train_path)
        os.replace(validation_tmp, validation_path)
    finally:
        train_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)

    manifest.write(output / "sources.json", require_pinned_revisions=require_pinned_revisions)
    stats = PreparationStats(
        input_documents=input_documents,
        train_documents=train_count,
        validation_documents=validation_count,
        duplicate_documents=duplicates,
        empty_documents=empty,
        by_source=counts,
    )
    _atomic_json_write(output / "stats.json", asdict(stats))
    return stats


def iter_jsonl_documents(path: str | Path) -> Iterator[Document]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                yield Document(
                    text=str(payload["text"]),
                    source=str(payload["source"]),
                    document_id=str(payload.get("id") or "") or None,
                )
            except KeyError as exc:
                raise ValueError(f"Malformed document at {path}:{line_number}") from exc


def iter_huggingface_documents(source: SourceSpec, *, streaming: bool = True) -> Iterator[Document]:
    """Lazily stream one pinned Hugging Face source."""
    if source.provider != "huggingface":
        raise ValueError(f"{source.name} is not a Hugging Face source")
    if source.revision in {"main", "master"}:
        raise ValueError("Resolve the source revision before downloading data")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to download Hugging Face corpora") from exc
    dataset = load_dataset(
        source.repository,
        source.subset,
        split=source.split,
        revision=source.revision,
        streaming=streaming,
    )
    for record in dataset:
        if not isinstance(record, Mapping):
            raise ValueError(f"Dataset {source.name} yielded a non-mapping record")
        yield _coerce_document(cast(Mapping[str, Any], record), source)


def download_wikimedia_dump(
    source: SourceSpec,
    output_dir: str | Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Materialize the one explicitly listed Wikimedia dump file atomically."""
    if source.provider != "wikimedia_dump":
        raise ValueError(f"{source.name} is not a Wikimedia dump source")
    if len(source.files) != 1:
        raise ValueError("A Wikimedia source must list exactly one pages-articles dump")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    from urllib.request import Request, urlopen

    source_file = source.files[0]
    revision_path = source.revision.replace("-", "")
    url = f"{source.repository.rstrip('/')}/{revision_path}/{source_file.path}"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / source_file.path
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "llm-pretrain/0.1 data preparation"})
    try:
        with urlopen(request) as response, temporary.open("wb") as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)
        expected_size = source_file.size_bytes
        if expected_size is not None and temporary.stat().st_size != expected_size:
            raise ValueError("Downloaded Wikimedia dump size does not match its manifest")
        if source_file.sha256 is not None and _file_sha256(temporary) != source_file.sha256:
            raise ValueError("Downloaded Wikimedia dump hash does not match its manifest")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def iter_wikimedia_documents(source: SourceSpec, dump_path: str | Path) -> Iterator[Document]:
    """Stream main-namespace, non-redirect pages from a Wikimedia XML bz2 dump."""
    if source.provider != "wikimedia_dump":
        raise ValueError(f"{source.name} is not a Wikimedia dump source")
    import bz2
    import xml.etree.ElementTree as element_tree

    path = Path(dump_path)
    with bz2.open(path, "rb") as stream:
        for _, element in element_tree.iterparse(stream, events=("end",)):
            if _xml_local_name(element.tag) != "page":
                continue
            direct_children = {_xml_local_name(child.tag): child for child in element}
            namespace = direct_children.get("ns")
            redirect = direct_children.get("redirect")
            revision = direct_children.get("revision")
            is_article = namespace is not None and namespace.text == "0"
            if not is_article or redirect is not None or revision is None:
                element.clear()
                continue
            text_element = next(
                (child for child in revision if _xml_local_name(child.tag) == "text"),
                None,
            )
            text = text_element.text if text_element is not None else None
            if text:
                identifier = direct_children.get("id")
                yield Document(
                    text=text,
                    source=source.name,
                    document_id=identifier.text if identifier is not None else None,
                )
            element.clear()


def allocate_token_quotas(
    total_tokens: int, sources: Sequence[SourceSpec] = DEFAULT_SOURCES
) -> dict[str, int]:
    """Allocate an exact integer token budget with largest-remainder rounding."""
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    total_weight = sum(source.token_weight for source in sources)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Source token weights must sum to 1.0")
    exact = [(source, total_tokens * source.token_weight) for source in sources]
    quotas = {source.name: math.floor(value) for source, value in exact}
    remainder = total_tokens - sum(quotas.values())
    order = sorted(exact, key=lambda item: (-(item[1] - math.floor(item[1])), item[0].name))
    for source, _ in order[:remainder]:
        quotas[source.name] += 1
    return quotas


@dataclass(frozen=True)
class PackedShard:
    path: str
    token_count: int
    sha256: str


@dataclass(frozen=True)
class ShardManifest:
    split: str
    sequence_length: int
    eos_id: int
    document_count: int
    encoded_token_count: int
    packed_token_count: int
    dropped_token_count: int
    shards: tuple[PackedShard, ...]
    source_token_counts: dict[str, int] = field(default_factory=dict)
    dtype: str = "uint32-le"
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path) -> ShardManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["shards"] = tuple(PackedShard(**item) for item in payload["shards"])
        return cls(**payload)


def _document_text(document: Document | str | Mapping[str, Any]) -> str:
    if isinstance(document, Document):
        return document.text
    if isinstance(document, str):
        return document
    text = document.get("text")
    if not isinstance(text, str):
        raise ValueError("Packed document mappings require a string `text` field")
    return text


def _document_source(document: Document | Mapping[str, Any]) -> str:
    if isinstance(document, Document):
        return document.source
    source = document.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("Mixed packed document mappings require a string `source` field")
    return source


def _validate_packing_options(
    split: str,
    sequence_length: int,
    shard_token_capacity: int,
    eos_id: int,
) -> None:
    if split not in {"train", "validation"}:
        raise ValueError("split must be `train` or `validation`")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if shard_token_capacity < sequence_length + 1:
        raise ValueError("shard_token_capacity must fit a sequence and its next-token label")
    if not 0 <= eos_id <= UINT32_MAX:
        raise ValueError("eos_id is outside uint32 range")


class _StreamingShardWriter:
    """Write fixed-capacity shards without retaining a shard-sized Python list."""

    def __init__(self, output: Path, split: str, capacity: int, numpy_module: Any) -> None:
        self.output = output
        self.split = split
        self.capacity = capacity
        self.np = numpy_module
        self.shards: list[PackedShard] = []
        self.packed_count = 0
        self.current_count = 0
        self._handle: Any | None = None
        self._temporary: Path | None = None

    def _open(self) -> None:
        if self._handle is not None:
            return
        index = len(self.shards)
        final_path = self.output / f"{self.split}-{index:05d}.bin"
        self._temporary = final_path.with_suffix(".bin.tmp")
        self._handle = self._temporary.open("wb")

    def write(self, tokens: Any) -> None:
        array = self.np.asarray(tokens, dtype="<u4").reshape(-1)
        position = 0
        while position < int(array.size):
            self._open()
            take = min(self.capacity - self.current_count, int(array.size) - position)
            array[position : position + take].tofile(self._handle)
            self.current_count += take
            position += take
            if self.current_count == self.capacity:
                self._finalize()

    def _finalize(self) -> None:
        if self._handle is None or self._temporary is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        final_path = self.output / f"{self.split}-{len(self.shards):05d}.bin"
        os.replace(self._temporary, final_path)
        self.shards.append(
            PackedShard(
                path=final_path.name,
                token_count=self.current_count,
                sha256=_file_sha256(final_path),
            )
        )
        self.packed_count += self.current_count
        self.current_count = 0
        self._handle = None
        self._temporary = None

    def finish(self, minimum_final_tokens: int) -> tuple[tuple[PackedShard, ...], int, int]:
        if self.current_count >= minimum_final_tokens:
            self._finalize()
            dropped = 0
        else:
            dropped = self.current_count
            self.abort()
        return tuple(self.shards), self.packed_count, dropped

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
        self.current_count = 0
        self._handle = None
        self._temporary = None


def pack_token_shards(
    documents: Iterable[Document | str | Mapping[str, Any]],
    tokenizer: TokenizerProtocol,
    output_dir: str | Path,
    *,
    split: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    shard_token_capacity: int = 16 * 1024 * 1024,
    eos_id: int | None = None,
) -> ShardManifest:
    """Append EOS per document and pack a continuous little-endian uint32 stream.

    Shards never mix splits.  A short final tail that cannot produce one
    ``sequence_length`` input plus its target token is dropped instead of
    padded.  Full shards retain a continuous stream and readers intentionally
    do not construct examples across physical shard boundaries.
    """
    resolved_eos = tokenizer.eos_id if eos_id is None else eos_id
    _validate_packing_options(split, sequence_length, shard_token_capacity, resolved_eos)

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - core project dependency
        raise RuntimeError("numpy is required to write packed token shards") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = _StreamingShardWriter(output, split, shard_token_capacity, np)
    document_count = encoded_count = 0
    try:
        for document in documents:
            ids = [int(token_id) for token_id in tokenizer.encode(_document_text(document))]
            ids.append(int(resolved_eos))
            if any(token_id < 0 or token_id > UINT32_MAX for token_id in ids):
                raise ValueError("Tokenizer emitted an id outside uint32 range")
            writer.write(ids)
            document_count += 1
            encoded_count += len(ids)
        shards, packed_count, dropped_count = writer.finish(sequence_length + 1)
    except BaseException:
        writer.abort()
        raise
    manifest = ShardManifest(
        split=split,
        sequence_length=sequence_length,
        eos_id=int(resolved_eos),
        document_count=document_count,
        encoded_token_count=encoded_count,
        packed_token_count=packed_count,
        dropped_token_count=dropped_count,
        shards=shards,
    )
    _atomic_json_write(output / f"{split}-shards.json", manifest.to_dict())
    return manifest


def pack_mixed_token_shards(
    documents: Iterable[Document | Mapping[str, Any]],
    tokenizer: TokenizerProtocol,
    output_dir: str | Path,
    *,
    split: str,
    source_token_quotas: Mapping[str, int],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    shard_token_capacity: int = 16 * 1024 * 1024,
    eos_id: int | None = None,
) -> ShardManifest:
    """Pack an exact deterministic source mix using disk-backed source buffers.

    Each source is capped at its requested token count after tokenization. Source
    streams are then interleaved with smooth weighted round-robin scheduling in
    ``sequence_length``-sized chunks. Temporary buffers live beside the output
    shards, so the operation needs bounded RAM even for billion-token corpora.
    """

    quotas = {str(name): int(value) for name, value in source_token_quotas.items()}
    if not quotas or any(not name or value <= 0 for name, value in quotas.items()):
        raise ValueError("source_token_quotas must contain positive counts")
    resolved_eos = tokenizer.eos_id if eos_id is None else eos_id
    _validate_packing_options(split, sequence_length, shard_token_capacity, resolved_eos)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - core project dependency
        raise RuntimeError("numpy is required to write packed token shards") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_order = tuple(quotas)
    counts = dict.fromkeys(source_order, 0)
    document_count = 0

    with tempfile.TemporaryDirectory(prefix=f".{split}-source-tokens-", dir=output) as raw_temp:
        temp_root = Path(raw_temp)
        temp_paths = {
            name: temp_root / f"source-{index:03d}.bin" for index, name in enumerate(source_order)
        }
        handles = {name: path.open("wb") for name, path in temp_paths.items()}
        try:
            for document in documents:
                if all(counts[name] >= quotas[name] for name in source_order):
                    break
                source = _document_source(document)
                if source not in quotas:
                    raise ValueError(f"Document belongs to unknown source {source!r}")
                remaining = quotas[source] - counts[source]
                if remaining <= 0:
                    continue
                ids = [int(token_id) for token_id in tokenizer.encode(_document_text(document))]
                ids.append(int(resolved_eos))
                if any(token_id < 0 or token_id > UINT32_MAX for token_id in ids):
                    raise ValueError("Tokenizer emitted an id outside uint32 range")
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    ids[-1] = int(resolved_eos)
                np.asarray(ids, dtype="<u4").tofile(handles[source])
                counts[source] += len(ids)
                document_count += 1
        finally:
            for handle in handles.values():
                handle.close()

        deficits = {
            name: quotas[name] - counts[name]
            for name in source_order
            if counts[name] < quotas[name]
        }
        if deficits:
            formatted = ", ".join(f"{name}={value}" for name, value in deficits.items())
            raise ValueError(f"Prepared corpus is short of requested tokens: {formatted}")

        writer = _StreamingShardWriter(output, split, shard_token_capacity, np)
        readers = {name: path.open("rb") for name, path in temp_paths.items()}
        remaining_by_source = dict(counts)
        total_tokens = sum(counts.values())
        weights = {name: counts[name] / total_tokens for name in source_order}
        credits = dict.fromkeys(source_order, 0.0)
        source_rank = {name: index for index, name in enumerate(source_order)}
        mix_chunk_tokens = max(
            sequence_length,
            min(shard_token_capacity, sequence_length * 64),
        )
        try:
            while any(remaining_by_source.values()):
                active = [name for name in source_order if remaining_by_source[name] > 0]
                for name in active:
                    credits[name] += weights[name]
                chosen = max(active, key=lambda name: (credits[name], -source_rank[name]))
                take = min(mix_chunk_tokens, remaining_by_source[chosen])
                chunk = np.fromfile(readers[chosen], dtype="<u4", count=take)
                if int(chunk.size) != take:
                    raise RuntimeError(f"Temporary token stream ended early for {chosen}")
                writer.write(chunk)
                remaining_by_source[chosen] -= take
                credits[chosen] -= take / mix_chunk_tokens
            shards, packed_count, dropped_count = writer.finish(sequence_length + 1)
        except BaseException:
            writer.abort()
            raise
        finally:
            for reader in readers.values():
                reader.close()

    manifest = ShardManifest(
        split=split,
        sequence_length=sequence_length,
        eos_id=int(resolved_eos),
        document_count=document_count,
        encoded_token_count=total_tokens,
        packed_token_count=packed_count,
        dropped_token_count=dropped_count,
        shards=shards,
        source_token_counts=counts,
    )
    _atomic_json_write(output / f"{split}-shards.json", manifest.to_dict())
    return manifest


@dataclass(frozen=True)
class DataCursor:
    epoch: int = 0
    shard_position: int = 0
    token_offset: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> DataCursor:
        return cls(
            epoch=int(state["epoch"]),
            shard_position=int(state["shard_position"]),
            token_offset=int(state["token_offset"]),
        )


@dataclass(frozen=True)
class TokenBatch(Mapping[str, Any]):
    input_ids: Any
    target_ids: Any
    cursor: DataCursor

    @property
    def labels(self) -> Any:
        """Unshifted labels for ``CausalLM``, which shifts them internally."""
        return self.input_ids

    def __getitem__(self, key: str) -> Any:
        if key == "input_ids":
            return self.input_ids
        if key == "labels":
            return self.input_ids
        if key == "target_ids":
            return self.target_ids
        if key == "cursor":
            return self.cursor
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "input_ids"
        yield "labels"
        yield "cursor"

    def __len__(self) -> int:
        return 3

    def to_torch(self, device: str | Any = "cpu") -> TokenBatch:
        """Convert numpy arrays to int64 torch tensors via a lazy import."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - core project dependency
            raise RuntimeError("torch is required to convert a token batch") from exc
        return TokenBatch(
            input_ids=torch.as_tensor(self.input_ids, dtype=torch.long, device=device),
            target_ids=torch.as_tensor(self.target_ids, dtype=torch.long, device=device),
            cursor=self.cursor,
        )


class MemmapTokenDataset:
    """Stateful, iterable next-token batch reader with exact resume support."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        batch_size: int,
        sequence_length: int | None = None,
        seed: int = DEFAULT_SEED,
        shuffle_shards: bool = True,
        repeat: bool = True,
        cursor: DataCursor | None = None,
        verify_hashes: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.manifest_path = Path(manifest_path)
        self.manifest = ShardManifest.read(self.manifest_path)
        self.root = self.manifest_path.parent
        self.batch_size = batch_size
        self.sequence_length = sequence_length or self.manifest.sequence_length
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.repeat = repeat
        self.cursor = cursor or DataCursor()
        self._validate_cursor(self.cursor)
        if verify_hashes:
            self.verify_shards()

    def _order(self, epoch: int) -> list[int]:
        order = list(range(len(self.manifest.shards)))
        if self.shuffle_shards:
            random.Random(self.seed + epoch).shuffle(order)
        return order

    def _validate_cursor(self, cursor: DataCursor) -> None:
        if cursor.epoch < 0 or cursor.shard_position < 0 or cursor.token_offset < 0:
            raise ValueError("Cursor fields cannot be negative")
        shard_count = len(self.manifest.shards)
        if shard_count and cursor.shard_position >= shard_count:
            raise ValueError("Cursor shard_position is outside the manifest")

    def verify_shards(self) -> None:
        for shard in self.manifest.shards:
            path = self.root / shard.path
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != shard.token_count * 4:
                raise ValueError(f"Shard size does not match manifest: {path}")
            if _file_sha256(path) != shard.sha256:
                raise ValueError(f"Shard hash does not match manifest: {path}")

    def state_dict(self) -> dict[str, int]:
        return self.cursor.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        cursor = DataCursor.from_state_dict(state)
        self._validate_cursor(cursor)
        self.cursor = cursor

    def _advance_shard(self) -> None:
        position = self.cursor.shard_position + 1
        epoch = self.cursor.epoch
        if position >= len(self.manifest.shards):
            if not self.repeat:
                raise StopIteration
            epoch += 1
            position = 0
        self.cursor = DataCursor(epoch=epoch, shard_position=position, token_offset=0)

    def _next_sequence(self) -> tuple[Any, Any]:
        if not self.manifest.shards:
            raise StopIteration
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - core project dependency
            raise RuntimeError("numpy is required to read packed token shards") from exc

        while True:
            order = self._order(self.cursor.epoch)
            shard = self.manifest.shards[order[self.cursor.shard_position]]
            path = self.root / shard.path
            tokens = np.memmap(path, mode="r", dtype="<u4", shape=(shard.token_count,))
            start = self.cursor.token_offset
            stop = start + self.sequence_length + 1
            if stop <= shard.token_count:
                inputs = np.asarray(tokens[start : stop - 1], dtype=np.int64).copy()
                targets = np.asarray(tokens[start + 1 : stop], dtype=np.int64).copy()
                self.cursor = DataCursor(
                    epoch=self.cursor.epoch,
                    shard_position=self.cursor.shard_position,
                    token_offset=start + self.sequence_length,
                )
                return inputs, targets
            self._advance_shard()

    def next_batch(self) -> TokenBatch:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - core project dependency
            raise RuntimeError("numpy is required to read packed token shards") from exc
        inputs = []
        targets = []
        starting_cursor = self.cursor
        try:
            for _ in range(self.batch_size):
                input_ids, target_ids = self._next_sequence()
                inputs.append(input_ids)
                targets.append(target_ids)
        except StopIteration:
            # ``drop_last`` semantics should not consume an unrecoverable
            # partial batch when a finite validation iterator is exhausted.
            self.cursor = starting_cursor
            raise
        return TokenBatch(
            input_ids=np.stack(inputs),
            target_ids=np.stack(targets),
            cursor=self.cursor,
        )

    def __iter__(self) -> Iterator[TokenBatch]:
        while True:
            try:
                yield self.next_batch()
            except StopIteration:
                return


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
