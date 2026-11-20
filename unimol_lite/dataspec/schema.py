"""SMS-DataSpec contract schemas used by the unimol_lite adapter.

The protocol (SMS-DataSpec ``spec_version = 1.0.0``) defines a container as
three parquet tables plus a JSON envelope.  This module carries the *contract*
part only: version constants, table names, the three column sets, hash/key mode
enumerations and the unimol_lite extension columns (``unimol_*``).

This is an independent implementation of the protocol: nothing here imports the
reference implementation.  Protocol compatibility is verified by tests
(round-trip, hashing vectors, batched reads, conformance).
"""

from __future__ import annotations

import polars as pl

#: Protocol version written into ``container_meta.json``.
SPEC_VERSION = "1.0.0"
#: Table-schema bundle version (column changes bump this).
SCHEMA_VERSION = "1"
#: Envelope file name inside a container directory.
CONTAINER_META_FILENAME = "container_meta.json"

TABLE_MOLECULES = "molecules"
TABLE_SPECTRA = "spectra"
TABLE_LINKER = "linker"
TABLE_NAMES: tuple[str, ...] = (TABLE_MOLECULES, TABLE_SPECTRA, TABLE_LINKER)

#: Row-level molecule key modes.
KEY_MODE_INCHIKEY = "inchikey"
KEY_MODE_SMILES_HASH = "smiles_hash"
MOLECULE_KEY_MODES: tuple[str, ...] = (KEY_MODE_INCHIKEY, KEY_MODE_SMILES_HASH)

#: Row-level spectrum hash modes.
HASH_MODE_SPLASH = "splash"
HASH_MODE_CONTENT_HASH = "content_hash"
SPECTRA_HASH_MODES: tuple[str, ...] = (HASH_MODE_SPLASH, HASH_MODE_CONTENT_HASH)

#: Container-level declarations additionally allow ``"mixed"``.
MOLECULE_KEY_MODES_CONTAINER: tuple[str, ...] = (*MOLECULE_KEY_MODES, "mixed")
SPECTRA_HASH_MODES_CONTAINER: tuple[str, ...] = (*SPECTRA_HASH_MODES, "mixed")

#: Declared training shuffle strategy (protocol section on shuffle semantics).
SHUFFLE_ARROW_MMAP_RANDOM_ROW = "arrow_mmap_random_row"
SHUFFLE_ROW_GROUP_SHUFFLE = "row_group_shuffle"
SHUFFLE_PRE_SHUFFLED_BLOCKS = "pre_shuffled_blocks"
SHUFFLE_STRATEGIES: tuple[str, ...] = (
    SHUFFLE_ARROW_MMAP_RANDOM_ROW,
    SHUFFLE_ROW_GROUP_SHUFFLE,
    SHUFFLE_PRE_SHUFFLED_BLOCKS,
)

# ---------------------------------------------------------------------------
# Contract schemas (column order == write order)
# ---------------------------------------------------------------------------

MOLECULES_SCHEMA = pl.Schema({
    "molecule_key": pl.String(),          # primary key, not null, unique
    "key_mode": pl.String(),              # row-level flag: inchikey | smiles_hash
    "inchikey": pl.String(),              # required when key_mode == inchikey
    "inchikey_skeleton": pl.String(),     # first 25 chars of inchikey
    "smiles": pl.String(),                # canonical SMILES
    "smiles_hash": pl.String(),           # required when key_mode == smiles_hash
    "inchi": pl.String(),                 # full InChI (stereo disambiguation)
    "name": pl.String(),
    "formula": pl.String(),
    "monoisotopic_mass": pl.Float64(),
})

SPECTRA_SCHEMA = pl.Schema({
    "spectrum_key": pl.String(),                 # primary key
    "hash_mode": pl.String(),                    # row-level flag: splash | content_hash
    "peaks_mz": pl.List(pl.Float64()),           # non-empty, same length as intensity
    "peaks_intensity": pl.List(pl.Float64()),    # non-empty, max > 0
    "intensity_normalized": pl.Boolean(),
    "ms_level": pl.Int64(),
    "precursor_mz": pl.Float64(),
    "precursor_type": pl.String(),
    "collision_energy": pl.Float64(),
    "ionization_mode": pl.String(),
})

LINKER_SCHEMA = pl.Schema({
    "spectrum_key": pl.String(),        # FK -> spectra.spectrum_key
    "molecule_key": pl.String(),        # FK -> molecules.molecule_key
    "source_library": pl.String(),
    "source_file": pl.String(),
    "acquisition_context": pl.String(),
})

TABLE_SCHEMAS: dict[str, pl.Schema] = {
    TABLE_MOLECULES: MOLECULES_SCHEMA,
    TABLE_SPECTRA: SPECTRA_SCHEMA,
    TABLE_LINKER: LINKER_SCHEMA,
}

# ---------------------------------------------------------------------------
# unimol_lite extension columns (namespace ``unimol_``; protocol allows extras)
# ---------------------------------------------------------------------------

#: Record kind flag; absent in foreign containers (treated as ``"smiles"``).
UNIMOL_KIND_COLUMN = "unimol_kind"
#: Atom symbols including hydrogens, as produced by featurization (docs/en/11-dataspec.md).
UNIMOL_ATOMS_COLUMN = "unimol_atoms"
#: Flattened ``x, y, z, ...`` coordinates, ``len == 3 * len(atoms)``.
UNIMOL_COORDS_COLUMN = "unimol_coords"
#: Provenance of the stored conformer: ``"3d"`` | ``"2d"``.
UNIMOL_COORD_MODE_COLUMN = "unimol_coord_mode"

UNIMOL_EXTENSION_SCHEMA = pl.Schema({
    UNIMOL_KIND_COLUMN: pl.String(),
    UNIMOL_ATOMS_COLUMN: pl.List(pl.String()),
    UNIMOL_COORDS_COLUMN: pl.List(pl.Float64()),
    UNIMOL_COORD_MODE_COLUMN: pl.String(),
})

#: Extension column names in write order.
UNIMOL_EXTENSION_COLUMNS: tuple[str, ...] = tuple(UNIMOL_EXTENSION_SCHEMA.names())

#: Record kinds understood by this adapter.
KIND_SMILES = "smiles"
KIND_PREGENERATED = "pregenerated"
KINDS: tuple[str, ...] = (KIND_SMILES, KIND_PREGENERATED)

#: Molecules table as written by this adapter: contract columns + ``unimol_*``.
MOLECULE_RECORD_SCHEMA = pl.Schema({**MOLECULES_SCHEMA, **UNIMOL_EXTENSION_SCHEMA})

#: Key columns that must exist in every readable container of each table.
REQUIRED_ROW_COLUMNS: dict[str, tuple[str, ...]] = {
    TABLE_MOLECULES: (
        "molecule_key",
        "key_mode",
        "inchikey",
        "inchikey_skeleton",
        "smiles",
        "smiles_hash",
        "inchi",
    ),
    TABLE_SPECTRA: (
        "spectrum_key",
        "hash_mode",
        "peaks_mz",
        "peaks_intensity",
        "intensity_normalized",
        "precursor_mz",
        "precursor_type",
        "collision_energy",
        "ionization_mode",
    ),
    TABLE_LINKER: ("spectrum_key", "molecule_key"),
}


def empty_table(table: str) -> pl.DataFrame:
    """Zero-row frame carrying the contract schema of ``table``."""
    return pl.DataFrame(schema=TABLE_SCHEMAS[table])
