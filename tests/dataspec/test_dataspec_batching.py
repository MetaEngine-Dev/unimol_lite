"""C6 — chunked write / batched read: batches concatenate to the whole table.

The contract is: writing lays down ``block_rows`` rows per parquet row group;
batched reading never lets a batch cross a row-group boundary; concatenating the
batches reproduces the table value by value.
"""

from __future__ import annotations

import polars as pl
import pyarrow.parquet as pq
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import big_molecules_frame

from unimol_lite.dataspec import (
    DEFAULT_BLOCK_ROWS,
    DataSpecContainer,
    write_table_chunks,
)
from unimol_lite.dataspec.schema import TABLE_MOLECULES, TABLE_SCHEMAS


class TestChunkedWrite:
    def test_block_rows_become_row_groups(self, make_container):
        bundle = make_container(name="blocked", molecules=big_molecules_frame(1000), block_rows=100)
        parquet_file = pq.ParquetFile(bundle.root / f"{TABLE_MOLECULES}.parquet")
        assert parquet_file.num_row_groups == 10
        assert bundle.meta.counts.molecules == 1000

    def test_default_block_size_constant(self):
        assert DEFAULT_BLOCK_ROWS == 250_000

    def test_rows_below_block_size_form_one_group(self, make_container):
        bundle = make_container(name="one-group", molecules=big_molecules_frame(37), block_rows=1000)
        parquet_file = pq.ParquetFile(bundle.root / f"{TABLE_MOLECULES}.parquet")
        assert parquet_file.num_row_groups == 1
        assert parquet_file.metadata.row_group(0).num_rows == 37

    def test_streaming_chunks_split_the_same_way(self, tmp_path):
        chunks = [big_molecules_frame(250) for _ in range(4)]
        path = tmp_path / "stream.parquet"
        total = write_table_chunks(
            path, TABLE_SCHEMAS[TABLE_MOLECULES], iter(chunks), block_rows=100
        )
        assert total == 1000
        parquet_file = pq.ParquetFile(path)
        heights = [parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups)]
        assert heights == [100, 100, 50] * 4

    def test_streaming_chunks_concatenate_to_input(self, tmp_path):
        chunks = [big_molecules_frame(120) for _ in range(3)]
        path = tmp_path / "concat-write.parquet"
        write_table_chunks(path, TABLE_SCHEMAS[TABLE_MOLECULES], iter(chunks), block_rows=64)
        assert pl.read_parquet(path).equals(pl.concat(chunks))

    def test_zero_row_stream_writes_an_empty_table(self, tmp_path):
        path = tmp_path / "empty.parquet"
        schema = TABLE_SCHEMAS[TABLE_MOLECULES]
        assert write_table_chunks(path, schema, iter([pl.DataFrame(schema=schema)]), block_rows=10) == 0
        assert path.is_file()
        assert pl.read_parquet(path).height == 0
        assert pl.read_parquet_schema(path) == schema

    def test_invalid_block_rows_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            write_table_chunks(tmp_path / "x.parquet", TABLE_SCHEMAS[TABLE_MOLECULES], iter([]), block_rows=0)

    def test_chunk_missing_contract_column_is_refused(self, tmp_path):
        from unimol_lite.dataspec import DataSpecError

        frame = big_molecules_frame(3).drop("inchi")
        with pytest.raises(DataSpecError):
            write_table_chunks(tmp_path / "bad.parquet", TABLE_SCHEMAS[TABLE_MOLECULES], iter([frame]))


class TestBatchedRead:
    def test_batches_respect_row_group_boundaries(self, make_container):
        bundle = make_container(name="rg-bound", molecules=big_molecules_frame(1000), block_rows=100)
        batches = list(bundle.container.iter_molecules_batches(250))
        # batch_rows (250) > block size (100): one batch per row group
        assert [batch.height for batch in batches] == [100] * 10

    def test_batches_split_inside_a_row_group(self, make_container):
        bundle = make_container(name="split", molecules=big_molecules_frame(1000), block_rows=1000)
        batches = list(bundle.container.iter_molecules_batches(37))
        heights = [batch.height for batch in batches]
        assert all(height <= 37 for height in heights)
        assert sum(heights) == 1000

    def test_batches_concatenate_to_the_whole_table(self, make_container):
        molecules = big_molecules_frame(400)
        bundle = make_container(name="concat", molecules=molecules, block_rows=120)
        joined = pl.concat(list(bundle.container.iter_molecules_batches(75)))
        assert joined.equals(molecules.cast(joined.schema))

    def test_row_group_iteration_equals_whole_table(self, make_container):
        molecules = big_molecules_frame(300)
        bundle = make_container(name="rg-iter", molecules=molecules, block_rows=70)
        joined = pl.concat(list(bundle.container.iter_table_row_groups(TABLE_MOLECULES)))
        assert joined.equals(molecules.cast(joined.schema))

    def test_empty_table_yields_no_batches(self, make_container):
        from unimol_lite.dataspec import empty_table

        bundle = make_container(
            name="empty-batches",
            molecules=empty_table(TABLE_MOLECULES),
            spectra=empty_table("spectra"),
            linker=empty_table("linker"),
        )
        assert list(bundle.container.iter_molecules_batches(10)) == []

    def test_invalid_batch_rows_is_refused(self, make_container):
        bundle = make_container()
        with pytest.raises(ValueError):
            list(bundle.container.iter_molecules_batches(0))

    def test_lazy_scan_matches_whole_table(self, make_container):
        bundle = make_container(name="scan")
        scanned = bundle.container.scan(TABLE_MOLECULES).collect()
        assert scanned.equals(bundle.container.molecules())


class TestKnownTable:
    def test_unknown_table_is_refused(self, make_container):
        from unimol_lite.dataspec import DataSpecError

        bundle = make_container(name="unknown")
        with pytest.raises(DataSpecError):
            DataSpecContainer(bundle.root).read("spectrums")
