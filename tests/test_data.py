from __future__ import annotations

import bz2
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_pretrain.data import (
    DEFAULT_SOURCES,
    DataCursor,
    Document,
    MemmapTokenDataset,
    SourceManifest,
    allocate_token_quotas,
    document_sha256,
    iter_wikimedia_documents,
    normalize_document,
    pack_mixed_token_shards,
    pack_token_shards,
    prepare_document_corpus,
    resolve_source_revisions,
    split_from_hash,
)


class CharacterTokenizer:
    eos_id = 3

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [10 + ord(character) for character in text]
        if add_bos:
            ids.insert(0, 2)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(token_id - 10) for token_id in ids if token_id >= 10)

    def id_to_piece(self, token_id: int) -> str:
        return str(token_id)


def pinned_manifest(seed: int = 7) -> SourceManifest:
    sources = tuple(
        replace(source, revision=("a" * 40 if source.revision == "main" else source.revision))
        for source in DEFAULT_SOURCES
    )
    return SourceManifest(sources=sources, seed=seed)


def test_default_source_mix_and_quota_rounding() -> None:
    assert [source.token_weight for source in DEFAULT_SOURCES] == [0.8, 0.1, 0.1]
    assert [source.requested_tokens for source in DEFAULT_SOURCES] == [
        1_600_000_000,
        200_000_000,
        200_000_000,
    ]
    assert allocate_token_quotas(101) == {
        "ultra_fineweb_zh": 81,
        "wikimedia_zh": 10,
        "fineweb_en": 10,
    }


def test_symbolic_revisions_are_resolved_and_manifest_round_trips(tmp_path: Path) -> None:
    resolved = resolve_source_revisions(
        DEFAULT_SOURCES,
        resolver=lambda repository, revision: "b" * 40,
    )
    manifest = SourceManifest(resolved, seed=11)
    path = manifest.write(tmp_path / "sources.json")
    loaded = SourceManifest.read(path)
    assert loaded == manifest
    assert resolved[0].revision == "b" * 40
    assert resolved[1].revision == "2026-08-01"


def test_normalization_hash_and_split_are_deterministic() -> None:
    left = "Ａ  B\r\n\r\n\r\nC  "  # noqa: RUF001
    right = "A B\n\nC"
    assert normalize_document(left) == right
    assert document_sha256(left) == document_sha256(right)
    digest = document_sha256(left)
    assert split_from_hash(digest, validation_fraction=0.2, seed=1) == split_from_hash(
        digest, validation_fraction=0.2, seed=1
    )


def test_prepare_corpus_exact_deduplicates_across_sources(tmp_path: Path) -> None:
    manifest = pinned_manifest()
    records = {
        "ultra_fineweb_zh": [
            Document("Ａ  B", "ignored", "first"),  # noqa: RUF001
            "unique 中文",
        ],
        "wikimedia_zh": ["A B"],
        "fineweb_en": ["  "],
    }
    stats = prepare_document_corpus(
        records,
        tmp_path,
        manifest=manifest,
        validation_fraction=0.5,
    )
    assert stats.input_documents == 4
    assert stats.duplicate_documents == 1
    assert stats.empty_documents == 1
    assert stats.train_documents + stats.validation_documents == 2

    rows = []
    for name in ("train.jsonl", "validation.jsonl"):
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        rows.extend(json.loads(line) for line in lines)
    assert len({row["sha256"] for row in rows}) == 2
    assert {row["text"] for row in rows} == {"A B", "unique 中文"}


def test_wikimedia_dump_stream_skips_redirects_and_non_articles(tmp_path: Path) -> None:
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>A</title><ns>0</ns><id>1</id><revision><text>article</text></revision></page>
    <page><title>B</title><ns>0</ns><id>2</id><redirect title="A"/>
      <revision><text>#REDIRECT</text></revision></page>
    <page><title>Talk:C</title><ns>1</ns><id>3</id><revision><text>talk</text></revision></page>
    </mediawiki>"""
    dump = tmp_path / "tiny.xml.bz2"
    dump.write_bytes(bz2.compress(xml))
    source = next(source for source in DEFAULT_SOURCES if source.name == "wikimedia_zh")
    documents = list(iter_wikimedia_documents(source, dump))
    assert documents == [Document(text="article", source="wikimedia_zh", document_id="1")]


def test_packer_writes_eos_continuous_uint32_shards(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    manifest = pack_token_shards(
        ["abcdefghi", "jklmnopqr"],
        tokenizer,
        tmp_path,
        split="train",
        sequence_length=4,
        shard_token_capacity=10,
    )
    assert manifest.document_count == 2
    assert manifest.encoded_token_count == 20
    assert manifest.packed_token_count == 20
    assert manifest.dropped_token_count == 0
    assert [shard.token_count for shard in manifest.shards] == [10, 10]
    first = np.fromfile(tmp_path / manifest.shards[0].path, dtype="<u4")
    assert first.dtype == np.dtype("uint32")
    assert first[-1] == tokenizer.eos_id


def test_memmap_batch_cursor_resumes_exactly(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    pack_token_shards(
        ["abcdefghi", "jklmnopqr"],
        tokenizer,
        tmp_path,
        split="train",
        sequence_length=4,
        shard_token_capacity=10,
    )
    manifest_path = tmp_path / "train-shards.json"
    reader = MemmapTokenDataset(
        manifest_path,
        batch_size=1,
        shuffle_shards=False,
        repeat=False,
        verify_hashes=True,
    )
    first = reader.next_batch()
    state = reader.state_dict()
    expected_second = reader.next_batch()

    resumed = MemmapTokenDataset(
        manifest_path,
        batch_size=1,
        shuffle_shards=False,
        repeat=False,
        cursor=DataCursor.from_state_dict(state),
    )
    actual_second = resumed.next_batch()
    assert first.input_ids.shape == (1, 4)
    assert first["labels"] is first.input_ids
    assert first["target_ids"] is first.target_ids
    np.testing.assert_array_equal(expected_second.input_ids, actual_second.input_ids)
    np.testing.assert_array_equal(expected_second.target_ids, actual_second.target_ids)
    assert expected_second.cursor == actual_second.cursor


def test_short_tail_is_dropped_instead_of_padded(tmp_path: Path) -> None:
    manifest = pack_token_shards(
        ["abc"],
        CharacterTokenizer(),
        tmp_path,
        split="validation",
        sequence_length=4,
        shard_token_capacity=8,
    )
    assert not manifest.shards
    assert manifest.dropped_token_count == 4
    reader = MemmapTokenDataset(
        tmp_path / "validation-shards.json",
        batch_size=1,
        repeat=False,
    )
    with pytest.raises(StopIteration):
        reader.next_batch()


def test_mixed_packer_enforces_exact_source_quotas_deterministically(tmp_path: Path) -> None:
    documents = [
        Document("a" * 20, "zh-web"),
        Document("b" * 20, "wiki"),
        Document("c" * 20, "en-web"),
    ]
    quotas = {"zh-web": 12, "wiki": 4, "en-web": 4}
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = pack_mixed_token_shards(
        documents,
        CharacterTokenizer(),
        first_dir,
        split="train",
        source_token_quotas=quotas,
        sequence_length=4,
        shard_token_capacity=10,
    )
    second = pack_mixed_token_shards(
        documents,
        CharacterTokenizer(),
        second_dir,
        split="train",
        source_token_quotas=quotas,
        sequence_length=4,
        shard_token_capacity=10,
    )

    assert first.source_token_counts == quotas
    assert first.encoded_token_count == 20
    assert first.packed_token_count == 20
    assert first.dropped_token_count == 0
    assert [shard.token_count for shard in first.shards] == [10, 10]
    for first_shard, second_shard in zip(first.shards, second.shards, strict=True):
        assert (first_dir / first_shard.path).read_bytes() == (
            second_dir / second_shard.path
        ).read_bytes()


def test_mixed_packer_reports_source_deficits_and_removes_temporary_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"wiki=10"):
        pack_mixed_token_shards(
            [Document("a" * 20, "zh-web")],
            CharacterTokenizer(),
            tmp_path,
            split="train",
            source_token_quotas={"zh-web": 10, "wiki": 10},
            sequence_length=4,
            shard_token_capacity=10,
        )
    assert not list(tmp_path.glob(".train-source-tokens-*"))
