"""C1 — three-table round-trip: write then read, value by value.

Covers null preservation, list columns, extension columns and the protocol's
requirement that readers accept both physical Arrow variants (``string`` /
``list`` and ``large_string`` / ``large_list``).
"""

from __future__ import annotations

import polars as pl
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import (
    INCHIKEY_ETHANOL,
    INCHIKEY_PHENOL,
    big_molecules_frame,
    molecules_frame,
    read_envelope_json,
    spectra_frame,
    write_small_type_table,
)

from unimol_lite.dataspec import (
    CONTAINER_META_FILENAME,
    TABLE_LINKER,
    TABLE_MOLECULES,
    TABLE_SPECTRA,
    DataSpecContainer,
    read_container_meta,
    read_table,
)


class TestRoundtrip:
    def test_molecules_roundtrip_value_by_value(self, make_container):
        bundle = make_container()
        read_back = read_table(bundle.root, TABLE_MOLECULES)
        assert read_back.height == bundle.molecules.height
        assert read_back.equals(bundle.molecules.cast(read_back.schema))
        assert read_back["molecule_key"].to_list() == [INCHIKEY_PHENOL, INCHIKEY_ETHANOL]
        assert read_back["key_mode"].to_list() == ["inchikey", "inchikey"]

    def test_null_values_survive(self, make_container):
        bundle = make_container()
        read_back = read_table(bundle.root, TABLE_MOLECULES)
        assert read_back["smiles_hash"].to_list() == [None, None]
        assert read_back["inchi"][1].startswith("InChI=1S/C2H6O")

    def test_list_columns_survive(self, make_container):
        spec = spectra_frame()
        bundle = make_container(name="lists", spectra=spec)
        read_back = read_table(bundle.root, TABLE_SPECTRA)
        assert read_back["peaks_mz"].to_list() == spec["peaks_mz"].to_list()
        assert read_back["peaks_intensity"].to_list() == spec["peaks_intensity"].to_list()
        assert read_back.schema["peaks_mz"] == pl.List(pl.Float64)

    def test_linker_roundtrip_keeps_order(self, make_container):
        bundle = make_container()
        read_back = read_table(bundle.root, TABLE_LINKER)
        assert read_back["spectrum_key"].to_list() == bundle.linker["spectrum_key"].to_list()
        assert read_back["molecule_key"].to_list() == bundle.linker["molecule_key"].to_list()

    def test_extension_column_is_preserved(self, make_container):
        bundle = make_container(name="extras", extra_column=True)
        read_back = read_table(bundle.root, TABLE_MOLECULES)
        assert "builder_note" in read_back.columns
        assert read_back["builder_note"].to_list() == ["fixture-row-1", "fixture-row-2"]

    def test_mixed_key_modes_roundtrip(self, make_container):
        bundle = make_container(name="mixed", mixed=True)
        read_back = read_table(bundle.root, TABLE_MOLECULES)
        assert read_back["key_mode"].to_list() == ["inchikey", "smiles_hash"]
        assert bundle.meta.hash_modes.molecules == "mixed"

    def test_envelope_file_is_written(self, make_container):
        bundle = make_container()
        assert (bundle.root / CONTAINER_META_FILENAME).is_file()
        payload = read_envelope_json(bundle.root)
        assert payload["spec_version"] == bundle.meta.spec_version
        assert payload["counts"] == {"molecules": 2, "spectra": 3, "linker": 4}
        assert read_container_meta(bundle.root) == bundle.meta

    def test_larger_molecule_table_roundtrip(self, make_container):
        mols = big_molecules_frame(500)
        bundle = make_container(name="big", molecules=mols, block_rows=64)
        container = DataSpecContainer(bundle.root)
        assert container.molecules().equals(mols.cast(container.molecules().schema))


class TestArrowTypeVariants:
    """Protocol: readers must accept ``utf8`` / ``list`` and their ``large_`` twins."""

    def test_writer_produces_large_variants(self, make_container):
        import pyarrow as pa
        import pyarrow.parquet as pq

        bundle = make_container()
        molecules_schema = pq.ParquetFile(bundle.root / f"{TABLE_MOLECULES}.parquet").schema_arrow
        spectra_schema = pq.ParquetFile(bundle.root / f"{TABLE_SPECTRA}.parquet").schema_arrow
        assert pa.types.is_large_string(molecules_schema.field("molecule_key").type)
        assert pa.types.is_large_list(spectra_schema.field("peaks_mz").type)

    def test_reader_accepts_small_type_variant(self, tmp_path, make_meta):
        import pyarrow.parquet as pq

        from unimol_lite.dataspec import HashModes, write_container
        from unimol_lite.dataspec.schema import LINKER_SCHEMA, SPECTRA_SCHEMA

        root = tmp_path / "small-types"
        mols = molecules_frame()
        write_container(
            root,
            mols,
            pl.DataFrame(schema=SPECTRA_SCHEMA),
            pl.DataFrame(schema=LINKER_SCHEMA),
            meta=make_meta(hash_modes=HashModes(spectra="content_hash", molecules="inchikey")),
        )
        # rewrite the molecules table with pyarrow small types
        write_small_type_table(root / f"{TABLE_MOLECULES}.parquet", mols)
        read_schema = pq.ParquetFile(root / f"{TABLE_MOLECULES}.parquet").schema_arrow
        assert str(read_schema.field("molecule_key").type) == "string"
        read_back = read_table(root, TABLE_MOLECULES)
        assert read_back["molecule_key"].to_list() == mols["molecule_key"].to_list()
        assert read_back.schema["molecule_key"] == pl.String
        assert DataSpecContainer(root).molecules().height == 2


@pytest.mark.parametrize("table", [TABLE_MOLECULES, TABLE_SPECTRA, TABLE_LINKER])
def test_zero_row_tables_still_exist(make_container, table):
    from unimol_lite.dataspec import empty_table

    bundle = make_container(
        name=f"empty-{table}",
        molecules=empty_table(TABLE_MOLECULES),
        spectra=empty_table(TABLE_SPECTRA),
        linker=empty_table(TABLE_LINKER),
    )
    assert (bundle.root / f"{table}.parquet").is_file()
    assert read_table(bundle.root, table).height == 0
