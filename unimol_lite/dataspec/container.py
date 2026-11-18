"""parquet-block container read/write (protocol: container + three tables).

Layout::

    <container-root>/
    ├── molecules.parquet
    ├── spectra.parquet
    ├── linker.parquet
    └── container_meta.json

Writing is chunked (``block_rows`` rows per parquet row group); reading is
batched (``iter_table_batches`` never lets a batch cross a row-group boundary)
and also offers lazy scans for random row access.

This is unimol_lite's own implementation of the protocol.  It writes what the
library can honestly produce: molecule tables (with ``unimol_*`` extension
columns), plus spectra/linker tables when the caller supplies them.  Spectrum
keys of ``hash_mode="content_hash"`` rows are verified on write; ``"splash"``
keys are accepted as opaque identifiers (SPLASH is not implemented here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import polars as pl
import pyarrow.parquet as pq

from .envelope import (
    ContainerMeta,
    CountsInfo,
    HashModes,
    assert_compatible,
    container_fingerprint,
    derive_mode,
    read_envelope,
    utc_now_iso,
)
from .errors import DataSpecError, DataSpecValidationError
from .hashes import spectrum_content_hash
from .schema import (
    CONTAINER_META_FILENAME,
    HASH_MODE_CONTENT_HASH,
    MOLECULE_KEY_MODES,
    SPEC_VERSION,
    SPECTRA_HASH_MODES,
    TABLE_LINKER,
    TABLE_MOLECULES,
    TABLE_NAMES,
    TABLE_SCHEMAS,
    TABLE_SPECTRA,
)

__all__ = [
    "DEFAULT_BLOCK_ROWS",
    "write_table_chunks",
    "write_container",
    "derive_hash_modes",
    "read_container_meta",
    "read_table",
    "DataSpecContainer",
]

#: Rows per parquet row group when writing (protocol recommends 250 000).
DEFAULT_BLOCK_ROWS = 250_000


def _table_path(root: Path, table: str) -> Path:
    if table not in TABLE_NAMES:
        raise DataSpecError(f"未知表: {table!r}（可选: {list(TABLE_NAMES)}）")
    return root / f"{table}.parquet"


def _prepare_chunk(chunk: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Check contract columns, cast them to contract dtypes, contract columns first."""
    missing = [name for name in schema.names() if name not in chunk.columns]
    if missing:
        raise DataSpecError(f"分块缺少契约列: {missing}")
    try:
        chunk = chunk.cast({name: dtype for name, dtype in schema.items()})
    except Exception as exc:  # polars raises InvalidOperation / ComputeError
        raise DataSpecError(f"契约列 dtype 转换失败: {exc}") from exc
    extras = [name for name in chunk.columns if name not in schema.names()]
    return chunk.select(list(schema.names()) + extras)


def write_table_chunks(
    path: str | Path,
    schema: pl.Schema,
    chunks: Iterable[pl.DataFrame],
    *,
    block_rows: int = DEFAULT_BLOCK_ROWS,
) -> int:
    """Stream chunks into one parquet file, one row group per ``block_rows`` slice.

    Returns the number of rows written.  A zero-row stream still creates a legal
    schema-only parquet file (the protocol requires all three tables to exist).
    """
    path = Path(path)
    if block_rows <= 0:
        raise ValueError(f"block_rows 必须为正数, got {block_rows}")
    writer: Optional[pq.ParquetWriter] = None
    final_schema: Optional[pl.Schema] = None
    total = 0
    try:
        for chunk in chunks:
            if chunk.height == 0:
                continue
            chunk = _prepare_chunk(chunk, schema)
            if final_schema is None:
                final_schema = chunk.schema
                writer = pq.ParquetWriter(path, chunk.to_arrow().schema)
            elif chunk.schema != final_schema:
                raise DataSpecError(f"分块 schema 漂移: {chunk.schema.names()} != {final_schema.names()}")
            table = chunk.to_arrow()
            for start in range(0, table.num_rows, block_rows):
                slice_ = table.slice(start, block_rows)
                # ``row_group_size`` bounds the group size of this write call; the
                # slice already being <= block_rows makes "one slice = one group"
                # unconditional.
                writer.write_table(slice_, row_group_size=block_rows)
                total += slice_.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.ParquetWriter(path, pl.DataFrame(schema=schema).to_arrow().schema).close()
    return total


def derive_hash_modes(
    molecules: Optional[pl.DataFrame] = None,
    spectra: Optional[pl.DataFrame] = None,
) -> HashModes:
    """Container-level ``hash_modes`` derived from the row-level flags."""
    molecule_modes: list[Optional[str]] = []
    if molecules is not None and "key_mode" in molecules.columns:
        molecule_modes = molecules["key_mode"].to_list()
    spectra_modes: list[Optional[str]] = []
    if spectra is not None and "hash_mode" in spectra.columns:
        spectra_modes = spectra["hash_mode"].to_list()
    return HashModes(
        spectra=derive_mode(spectra_modes, legal=SPECTRA_HASH_MODES, default=SPECTRA_HASH_MODES[0]),
        molecules=derive_mode(molecule_modes, legal=MOLECULE_KEY_MODES, default=MOLECULE_KEY_MODES[0]),
    )


def _assert_declared_modes(meta: ContainerMeta, frames: dict[str, pl.DataFrame]) -> None:
    """Refuse to write an envelope whose declared modes contradict the rows."""
    spectra = frames[TABLE_SPECTRA]
    if spectra.height:
        if "hash_mode" not in spectra.columns:
            raise DataSpecError(f"分块缺少契约列: ['hash_mode']（{TABLE_SPECTRA}）")
        actual = derive_mode(
            spectra["hash_mode"].drop_nulls().to_list(),
            legal=SPECTRA_HASH_MODES,
            default=meta.hash_modes.spectra,
        )
        if actual != meta.hash_modes.spectra:
            raise DataSpecError(
                f"envelope 声明 spectra={meta.hash_modes.spectra!r}，数据行级实际 {actual!r}——虚报写不进容器"
            )
    molecules = frames[TABLE_MOLECULES]
    if molecules.height:
        if "key_mode" not in molecules.columns:
            raise DataSpecError(f"分块缺少契约列: ['key_mode']（{TABLE_MOLECULES}）")
        actual = derive_mode(
            molecules["key_mode"].drop_nulls().to_list(),
            legal=MOLECULE_KEY_MODES,
            default=meta.hash_modes.molecules,
        )
        if actual != meta.hash_modes.molecules:
            raise DataSpecError(
                f"envelope 声明 molecules={meta.hash_modes.molecules!r}，数据行级实际 {actual!r}——虚报写不进容器"
            )


def _assert_content_hash_rows(spectra: pl.DataFrame) -> None:
    """Verify ``content_hash`` spectrum keys against the stored peaks.

    ``splash`` rows are accepted as-is: this library cannot recompute SPLASH, so
    those keys are the producer's responsibility (conformance marks them
    unverified).
    """
    if not spectra.height or "hash_mode" not in spectra.columns:
        return
    for row in spectra.iter_rows(named=True):
        if row["hash_mode"] != HASH_MODE_CONTENT_HASH:
            continue
        expected = spectrum_content_hash(
            row["peaks_mz"],
            row["peaks_intensity"],
            precursor_mz=row["precursor_mz"],
            precursor_type=row["precursor_type"],
            ionization_mode=row["ionization_mode"],
            collision_energy=row["collision_energy"],
        )
        if row["spectrum_key"] != expected:
            raise DataSpecError(
                f"content_hash 谱图键与内容不一致（存储 {row['spectrum_key']!r} != 重算 {expected!r}）"
                "——键必须与内容绑定"
            )


def write_container(
    root: str | Path,
    molecules: Optional[pl.DataFrame] = None,
    spectra: Optional[pl.DataFrame] = None,
    linker: Optional[pl.DataFrame] = None,
    *,
    meta: ContainerMeta,
    block_rows: int = DEFAULT_BLOCK_ROWS,
) -> ContainerMeta:
    """Write a whole container: three tables plus a finalised envelope.

    ``counts`` / ``created_at`` / ``fingerprint`` on the passed ``meta`` are
    recomputed from what is actually written.  Returns the finalised meta.
    """
    major = meta.spec_version.split(".")[0]
    if major != SPEC_VERSION.split(".")[0]:
        raise DataSpecValidationError(
            f"spec 大版本不支持: {meta.spec_version}（本适配器写 {SPEC_VERSION}）——拒绝写入"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    frames = {
        TABLE_MOLECULES: molecules if molecules is not None else pl.DataFrame(schema=TABLE_SCHEMAS[TABLE_MOLECULES]),
        TABLE_SPECTRA: spectra if spectra is not None else pl.DataFrame(schema=TABLE_SCHEMAS[TABLE_SPECTRA]),
        TABLE_LINKER: linker if linker is not None else pl.DataFrame(schema=TABLE_SCHEMAS[TABLE_LINKER]),
    }
    _assert_declared_modes(meta, frames)
    _assert_content_hash_rows(frames[TABLE_SPECTRA])

    counts: dict[str, int] = {}
    for table in TABLE_NAMES:
        counts[table] = write_table_chunks(
            _table_path(root, table),
            TABLE_SCHEMAS[table],
            iter([frames[table]]),
            block_rows=block_rows,
        )
    finalized = meta.with_updates(
        counts=CountsInfo(**counts),
        created_at=meta.created_at or utc_now_iso(),
    )
    finalized = finalized.with_updates(fingerprint=container_fingerprint(finalized))
    (root / CONTAINER_META_FILENAME).write_text(
        json.dumps(finalized.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return finalized


def read_container_meta(root: str | Path) -> ContainerMeta:
    """Read and validate the envelope (fail-closed; see :func:`read_envelope`)."""
    return read_envelope(root)


def read_table(root: str | Path, table: str) -> pl.DataFrame:
    """Read a whole table (small/verification use; batch APIs for training)."""
    frame = pl.read_parquet(_table_path(Path(root), table))
    return frame


class DataSpecContainer:
    """Read side of a container: envelope, whole-table reads, scans, batches.

    Construction validates the envelope and the presence of the three table
    files; a container that fails either is refused.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta = read_container_meta(self.root)
        for table in TABLE_NAMES:
            path = _table_path(self.root, table)
            if not path.is_file():
                raise DataSpecValidationError(f"表文件缺失: {path}")

    # -- basics --------------------------------------------------------------
    @property
    def fingerprint(self) -> Optional[str]:
        return self.meta.fingerprint

    def table_path(self, table: str) -> Path:
        return _table_path(self.root, table)

    def read(self, table: str) -> pl.DataFrame:
        return read_table(self.root, table)

    def molecules(self) -> pl.DataFrame:
        return self.read(TABLE_MOLECULES)

    def spectra(self) -> pl.DataFrame:
        return self.read(TABLE_SPECTRA)

    def linker(self) -> pl.DataFrame:
        return self.read(TABLE_LINKER)

    # -- random row access (lazy scan / predicate pushdown) -------------------
    def scan(self, table: str) -> pl.LazyFrame:
        return pl.scan_parquet(self.table_path(table))

    # -- row-group level iteration -------------------------------------------
    def iter_table_row_groups(self, table: str) -> Iterator[pl.DataFrame]:
        """Yield one frame per parquet row group, in file order."""
        parquet_file = pq.ParquetFile(self.table_path(table))
        for index in range(parquet_file.num_row_groups):
            yield pl.from_arrow(parquet_file.read_row_group(index))

    # -- batched iteration (respects row-group boundaries) --------------------
    def iter_table_batches(self, table: str, batch_rows: int) -> Iterator[pl.DataFrame]:
        """Yield frames of at most ``batch_rows`` rows; a batch never spans row groups."""
        if batch_rows <= 0:
            raise ValueError(f"batch_rows 必须为正数, got {batch_rows}")
        for group in self.iter_table_row_groups(table):
            for start in range(0, group.height, batch_rows):
                yield group.slice(start, batch_rows)

    def iter_molecules_batches(self, batch_rows: int) -> Iterator[pl.DataFrame]:
        return self.iter_table_batches(TABLE_MOLECULES, batch_rows)

    # -- mixing guard ---------------------------------------------------------
    def assert_compatible_with(self, other: "ContainerMeta | str") -> None:
        """Raise :class:`DataSpecMismatchError` if contract fingerprints differ."""
        assert_compatible(self.meta, other)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DataSpecContainer({str(self.root)!r}, spec={self.meta.spec_version}, counts={self.meta.counts})"
