"""unimol_lite record kinds on top of the container: payloads, round-trip, items.

Beyond the protocol's own contracts, these tests pin what is specific to this
library: the ``smiles`` and ``pregenerated`` kinds, the identity columns the
builder is responsible for, and the fact that reading a pregenerated record needs
no RDKit embedding (so it reproduces the same tensors).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import INCHIKEY_ETHANOL, INCHIKEY_PHENOL, spectra_frame
from unimol_lite.dataspec import (
    KIND_PREGENERATED,
    KIND_SMILES,
    KINDS,
    DataSpecValidationError,
    build_record,
    frame_from_records,
    iter_items,
    iter_records,
    molecule_identity,
    record_to_item,
    validate_container,
    write_records,
)
from unimol_lite.dataspec.schema import (
    MOLECULE_RECORD_SCHEMA,
    TABLE_MOLECULES,
    UNIMOL_ATOMS_COLUMN,
    UNIMOL_COORD_MODE_COLUMN,
    UNIMOL_COORDS_COLUMN,
    UNIMOL_KIND_COLUMN,
)


@pytest.fixture(scope="module")
def ethanol_record() -> dict:
    return build_record("CCO")


@pytest.fixture(scope="module")
def pregenerated_record() -> dict:
    return build_record("CCO", kind=KIND_PREGENERATED, timeout=30.0)


@pytest.fixture(scope="module")
def pregenerated_phenol() -> dict:
    """A second molecule, so writes can mix both kinds without duplicate keys."""
    return build_record("c1ccccc1O", kind=KIND_PREGENERATED, timeout=30.0)


class TestIdentity:
    def test_inchikey_is_the_default_key(self, ethanol_record):
        assert ethanol_record["key_mode"] == "inchikey"
        assert ethanol_record["molecule_key"] == INCHIKEY_ETHANOL
        assert ethanol_record["inchikey"] == INCHIKEY_ETHANOL
        assert ethanol_record["inchikey_skeleton"] == INCHIKEY_ETHANOL[:25]
        assert ethanol_record["smiles"] == "CCO"

    def test_inchikey_mode_leaves_smiles_hash_empty(self, ethanol_record):
        assert ethanol_record["smiles_hash"] is None

    def test_canonicalisation_happens_in_the_builder(self):
        non_canonical = build_record("OCC")
        canonical = build_record("CCO")
        assert non_canonical["smiles"] == canonical["smiles"] == "CCO"
        assert non_canonical["molecule_key"] == canonical["molecule_key"]

    def test_metadata_columns_are_filled(self, ethanol_record):
        assert ethanol_record["formula"] == "C2H6O"
        assert ethanol_record["monoisotopic_mass"] == pytest.approx(46.0419, abs=1e-3)
        assert ethanol_record["inchi"].startswith("InChI=1S/C2H6O")

    def test_inchi_can_be_skipped(self):
        record = molecule_identity("CCO", with_inchi=False)
        assert record["inchi"] is None

    def test_invalid_smiles_is_refused(self):
        with pytest.raises(ValueError):
            molecule_identity("not-a-smiles")

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            build_record("CCO", kind="tensors")


class TestKinds:
    def test_smiles_kind_has_no_payload(self, ethanol_record):
        assert ethanol_record[UNIMOL_KIND_COLUMN] == KIND_SMILES
        assert ethanol_record[UNIMOL_ATOMS_COLUMN] is None
        assert ethanol_record[UNIMOL_COORDS_COLUMN] is None
        assert ethanol_record[UNIMOL_COORD_MODE_COLUMN] is None

    def test_pregenerated_kind_carries_atoms_and_coordinates(self, pregenerated_record):
        atoms = pregenerated_record[UNIMOL_ATOMS_COLUMN]
        coords = pregenerated_record[UNIMOL_COORDS_COLUMN]
        assert pregenerated_record[UNIMOL_KIND_COLUMN] == KIND_PREGENERATED
        assert atoms and len(atoms) == 9  # CCO with hydrogens: C2H6O
        assert len(coords) == 3 * len(atoms)
        assert pregenerated_record[UNIMOL_COORD_MODE_COLUMN] in ("3d", "2d")

    def test_pregenerated_payload_is_reproducible(self, make_container):
        first = build_record("CCO", kind=KIND_PREGENERATED, timeout=30.0)
        second = build_record("CCO", kind=KIND_PREGENERATED, timeout=30.0)
        assert first[UNIMOL_ATOMS_COLUMN] == second[UNIMOL_ATOMS_COLUMN]
        assert first[UNIMOL_COORDS_COLUMN] == second[UNIMOL_COORDS_COLUMN]

    def test_kind_list_is_explicit(self):
        assert KINDS == ("smiles", "pregenerated")


class TestFrames:
    def test_frame_has_contract_columns_then_extension(self, ethanol_record):
        frame = frame_from_records([ethanol_record])
        assert frame.columns == list(MOLECULE_RECORD_SCHEMA.names())
        assert frame.schema[UNIMOL_ATOMS_COLUMN] == pl.List(pl.String)
        assert frame.schema[UNIMOL_COORDS_COLUMN] == pl.List(pl.Float64)

    def test_empty_frame_keeps_the_schema(self):
        frame = frame_from_records([])
        assert frame.height == 0
        assert frame.schema == MOLECULE_RECORD_SCHEMA

    def test_record_without_key_is_refused(self):
        with pytest.raises(DataSpecValidationError):
            frame_from_records([{"smiles": "CCO"}])

    def test_pregenerated_payload_mismatch_is_refused(self, pregenerated_record):
        broken = dict(pregenerated_record)
        broken[UNIMOL_COORDS_COLUMN] = broken[UNIMOL_COORDS_COLUMN][:-1]
        with pytest.raises(DataSpecValidationError):
            frame_from_records([broken])


class TestWriteAndRead:
    def test_roundtrip_two_kinds(self, tmp_path, ethanol_record, pregenerated_phenol):
        root = tmp_path / "records.msdata"
        meta = write_records(root, [ethanol_record, pregenerated_phenol], shuffle_strategy="row_group_shuffle")
        assert meta.counts.molecules == 2
        assert validate_container(root).ok
        records = list(iter_records(root))
        assert [record[UNIMOL_KIND_COLUMN] for record in records] == [KIND_SMILES, KIND_PREGENERATED]
        assert records[1][UNIMOL_COORDS_COLUMN] == pregenerated_phenol[UNIMOL_COORDS_COLUMN]

    def test_kind_filter(self, tmp_path, ethanol_record, pregenerated_phenol):
        root = tmp_path / "filter.msdata"
        write_records(root, [ethanol_record, pregenerated_phenol])
        only_pre = list(iter_records(root, kind=KIND_PREGENERATED))
        assert len(only_pre) == 1
        assert only_pre[0][UNIMOL_KIND_COLUMN] == KIND_PREGENERATED

    def test_projection(self, tmp_path, ethanol_record):
        root = tmp_path / "projection.msdata"
        write_records(root, [ethanol_record])
        records = list(iter_records(root, columns=("smiles", "molecule_key")))
        assert set(records[0]) == {"smiles", "molecule_key", UNIMOL_KIND_COLUMN}

    def test_foreign_container_reads_as_smiles_records(self, tmp_path, make_container):
        """A container without ``unimol_*`` columns is read as plain SMILES records."""
        bundle = make_container(name="foreign")
        records = list(iter_records(bundle.root))
        assert [record[UNIMOL_KIND_COLUMN] for record in records] == [KIND_SMILES, KIND_SMILES]
        assert [record["molecule_key"] for record in records] == [INCHIKEY_PHENOL, INCHIKEY_ETHANOL]

    def test_kind_filter_on_container_without_kind_column_is_refused(self, tmp_path, make_container):
        """P3-b：kind=pregenerated 过滤缺 unimol_kind 列的容器 → 显式 ValueError（不静默零行）。"""
        bundle = make_container(name="foreign-pregenerated")
        with pytest.raises(ValueError, match="unimol_kind"):
            list(iter_records(bundle.root, kind=KIND_PREGENERATED))
        # kind="smiles"（默认语义）在缺列容器上仍合法——全部行默认 smiles
        assert len(list(iter_records(bundle.root, kind=KIND_SMILES))) == 2

    def test_duplicate_keys_are_refused_on_write(self, tmp_path, ethanol_record):
        root = tmp_path / "dup.msdata"
        with pytest.raises(DataSpecValidationError):
            write_records(root, [ethanol_record, dict(ethanol_record)])

    def test_shuffle_declaration_is_recorded(self, tmp_path, ethanol_record):
        root = tmp_path / "shuffle.msdata"
        meta = write_records(root, [ethanol_record], shuffle_strategy="arrow_mmap_random_row", metric="dot")
        assert meta.shuffle_strategy == "arrow_mmap_random_row"
        assert meta.metric == "dot"

    def test_spectra_can_be_passed_through(self, tmp_path, ethanol_record):
        root = tmp_path / "with-spectra.msdata"
        write_records(root, [ethanol_record], spectra=spectra_frame())
        assert validate_container(root).ok
        assert pl.read_parquet(root / "spectra.parquet").height == 3


class TestRecordToItem:
    def test_smiles_record_is_featurized(self, ethanol_record):
        item, meta = record_to_item(ethanol_record)
        assert meta["status"] == "ok"
        assert meta["record_kind"] == KIND_SMILES
        assert meta["molecule_key"] == INCHIKEY_ETHANOL
        assert item["src_tokens"].shape[0] == 5  # CLS + 3 heavy atoms + SEP
        assert item["src_coord"].shape == (5, 3)

    def test_pregenerated_record_needs_no_embedding(self, pregenerated_record):
        first, meta = record_to_item(pregenerated_record)
        second, _ = record_to_item(pregenerated_record)
        assert meta["status"] == "ok"
        assert meta["record_kind"] == KIND_PREGENERATED
        for key in ("src_tokens", "src_coord", "src_distance", "src_edge_type"):
            assert (first[key] == second[key]).all(), key

    def test_pregenerated_matches_direct_generation(self, pregenerated_record):
        from unimol_lite.data.prepare_unimol_item import generate_unimolitem

        item, _ = record_to_item(pregenerated_record)
        expected = generate_unimolitem(
            atoms=np.asarray(pregenerated_record[UNIMOL_ATOMS_COLUMN]),
            coordinates=np.asarray(pregenerated_record[UNIMOL_COORDS_COLUMN]).reshape(-1, 3),
            random_seed=42,
            max_atoms=256,
        )
        for key in ("src_tokens", "src_coord", "src_distance", "src_edge_type"):
            assert (item[key] == expected[key]).all(), key

    def test_broken_payload_is_reported_not_raised(self, pregenerated_record):
        broken = dict(pregenerated_record)
        broken[UNIMOL_COORDS_COLUMN] = None
        item, meta = record_to_item(broken)
        assert item is None
        assert meta["status"] == "invalid"
        assert meta["reason"] == "pregenerated_payload_missing"

    def test_missing_smiles_is_reported(self):
        item, meta = record_to_item({"molecule_key": "x", UNIMOL_KIND_COLUMN: KIND_SMILES})
        assert item is None
        assert meta["reason"] == "smiles_missing"


class TestIterItems:
    def test_stream_yields_record_item_and_meta(self, tmp_path, ethanol_record, pregenerated_phenol):
        root = tmp_path / "stream.msdata"
        write_records(root, [ethanol_record, pregenerated_phenol])
        triples = list(iter_items(root))
        assert len(triples) == 2
        for record, item, meta in triples:
            assert item is not None
            assert meta["status"] == "ok"
            assert meta["molecule_key"] == record["molecule_key"]

    def test_batch_rows_do_not_change_the_stream(self, tmp_path, ethanol_record, pregenerated_phenol):
        root = tmp_path / "batched.msdata"
        write_records(root, [ethanol_record, pregenerated_phenol], block_rows=1)
        keys_small = [record["molecule_key"] for record, _, _ in iter_items(root, batch_rows=1)]
        keys_large = [record["molecule_key"] for record, _, _ in iter_items(root, batch_rows=1000)]
        assert keys_small == keys_large


def test_unimol_extension_columns_are_written(tmp_path, ethanol_record):
    root = tmp_path / "ext.msdata"
    write_records(root, [ethanol_record])
    frame = pl.read_parquet(root / f"{TABLE_MOLECULES}.parquet")
    assert frame[UNIMOL_KIND_COLUMN].to_list() == [KIND_SMILES]
    assert UNIMOL_ATOMS_COLUMN in frame.columns
