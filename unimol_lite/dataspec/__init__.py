"""SMS-DataSpec data adapter (parquet-block containers) for unimol_lite.

SMS-DataSpec is the standard spectrum/molecule data protocol: a container is a
directory holding three parquet tables (``molecules`` / ``spectra`` / ``linker``)
plus a JSON envelope (``container_meta.json``).  This package is unimol_lite's
**own** implementation of that protocol — it does not import any other library's
adapter; compatibility is what the tests check (round-trip, hash vectors, batched
reads, conformance).

Scope of this adapter
---------------------

- **Molecule domain.**  Records are molecules (kinds ``smiles`` and
  ``pregenerated``; see :mod:`unimol_lite.dataspec.records`).  The ``spectra``
  and ``linker`` tables are read and passed through, but this library neither
  derives nor verifies SPLASH keys (it is not a spectrum tool): ``splash`` keys
  are opaque identifiers, ``content_hash`` keys are computed and verified.
- **Pretraining LMDB is out of scope.**  The pretraining LMDB layout follows the
  upstream Uni-Mol project and is exempt from this protocol.
- **Optional backend.**  ``polars`` and ``pyarrow`` are needed only here
  (``pip install "unimol_lite[dataspec]"``); the rest of the library works
  without them.

Quick use::

    from unimol_lite.dataspec import build_record, write_records, iter_items

    records = [build_record("CCO", kind="pregenerated")]
    write_records("demo.msdata", records)
    for record, item, meta in iter_items("demo.msdata"):
        ...
"""

from __future__ import annotations

try:  # backend guard: a clear message instead of a raw ImportError deep inside
    import polars as _polars  # noqa: F401
    import pyarrow as _pyarrow  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "unimol_lite.dataspec requires polars and pyarrow; "
        'install the optional extra: pip install "unimol_lite[dataspec]"'
    ) from exc

from .conformance import ConformanceIssue, ConformanceReport, validate_container
from .container import (
    DEFAULT_BLOCK_ROWS,
    DataSpecContainer,
    derive_hash_modes,
    read_container_meta,
    read_table,
    write_container,
    write_table_chunks,
)
from .envelope import (
    BuilderInfo,
    ContainerMeta,
    CountsInfo,
    HashModes,
    SourceInfo,
    SourceReference,
    assert_compatible,
    container_fingerprint,
)
from .errors import DataSpecError, DataSpecMismatchError, DataSpecValidationError
from .hashes import inchikey_skeleton, smiles_hash, spectrum_content_hash, validate_peaks
from .records import (
    DEFAULT_RECORD_BATCH_ROWS,
    KIND_PREGENERATED,
    KIND_SMILES,
    KINDS,
    build_record,
    frame_from_records,
    iter_items,
    iter_records,
    molecule_identity,
    record_to_item,
    write_records,
)
from .schema import (
    CONTAINER_META_FILENAME,
    MOLECULE_KEY_MODES,
    SCHEMA_VERSION,
    SPEC_VERSION,
    SPECTRA_HASH_MODES,
    TABLE_LINKER,
    TABLE_MOLECULES,
    TABLE_NAMES,
    TABLE_SPECTRA,
    empty_table,
)

__all__ = [
    # protocol constants
    "SPEC_VERSION",
    "SCHEMA_VERSION",
    "CONTAINER_META_FILENAME",
    "TABLE_NAMES",
    "TABLE_MOLECULES",
    "TABLE_SPECTRA",
    "TABLE_LINKER",
    "MOLECULE_KEY_MODES",
    "SPECTRA_HASH_MODES",
    "empty_table",
    # container I/O
    "DEFAULT_BLOCK_ROWS",
    "DataSpecContainer",
    "write_container",
    "write_table_chunks",
    "read_container_meta",
    "read_table",
    "derive_hash_modes",
    # envelope / contracts
    "ContainerMeta",
    "CountsInfo",
    "HashModes",
    "SourceInfo",
    "SourceReference",
    "BuilderInfo",
    "container_fingerprint",
    "assert_compatible",
    # hashes
    "smiles_hash",
    "inchikey_skeleton",
    "spectrum_content_hash",
    "validate_peaks",
    # molecule records (unimol kinds)
    "KINDS",
    "KIND_SMILES",
    "KIND_PREGENERATED",
    "DEFAULT_RECORD_BATCH_ROWS",
    "molecule_identity",
    "build_record",
    "frame_from_records",
    "write_records",
    "iter_records",
    "record_to_item",
    "iter_items",
    # conformance
    "validate_container",
    "ConformanceIssue",
    "ConformanceReport",
    # errors
    "DataSpecError",
    "DataSpecValidationError",
    "DataSpecMismatchError",
]
