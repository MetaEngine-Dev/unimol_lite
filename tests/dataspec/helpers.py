"""Shared helpers for the SMS-DataSpec adapter tests (no fixtures here).

Fixtures live in ``tests/dataspec/conftest.py``; test modules guard themselves
with ``pytest.importorskip("polars")`` so the suite still runs without the
``dataspec`` extra installed.

Self-contained: every container is built under ``tmp_path`` from small synthetic
data.  Nothing here imports another library's adapter — the fixtures use *this*
implementation's hashing for row keys (which is what round-trip / reconciliation
tests are about), while the hash algorithms themselves are locked by the external
literals in ``test_dataspec_hashes.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from unimol_lite.dataspec import (
    SCHEMA_VERSION,
    SPEC_VERSION,
    ContainerMeta,
    DataSpecContainer,
    empty_table,
    inchikey_skeleton,
    smiles_hash,
    spectrum_content_hash,
)
from unimol_lite.dataspec.schema import LINKER_SCHEMA

#: Real InChIKeys for the SMILES used below (ethanol / phenol).
INCHIKEY_ETHANOL = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
INCHIKEY_PHENOL = "ISWSIDIOOBJBQZ-UHFFFAOYSA-N"

#: Peak sets used by the spectra fixtures (all degenerate-free).
PEAK_SETS: list[tuple[list[float], list[float]]] = [
    ([100.0, 200.5, 350.25], [100.0, 40.0, 12.5]),
    ([50.1, 101.3], [80.0, 100.0]),
    ([310.0], [5.0]),
]


def molecules_frame(*, mixed: bool = False, extra_column: bool = False, with_inchi: bool = True) -> pl.DataFrame:
    """Two molecules: phenol (inchikey) and ethanol (inchikey, or smiles_hash when mixed)."""
    if mixed:
        second_mode = "smiles_hash"
        second_key = smiles_hash("CCO")
        second_inchikey = None
        second_skeleton = None
    else:
        second_mode = "inchikey"
        second_key = INCHIKEY_ETHANOL
        second_inchikey = INCHIKEY_ETHANOL
        second_skeleton = inchikey_skeleton(INCHIKEY_ETHANOL)
    data: dict[str, object] = {
        "molecule_key": [INCHIKEY_PHENOL, second_key],
        "key_mode": ["inchikey", second_mode],
        "inchikey": [INCHIKEY_PHENOL, second_inchikey],
        "inchikey_skeleton": [inchikey_skeleton(INCHIKEY_PHENOL), second_skeleton],
        "smiles": ["C1=CC=C(C=C1)O", "CCO"],
        "smiles_hash": [None, smiles_hash("CCO") if mixed else None],
        "inchi": ["InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"]
        if with_inchi
        else [None, None],
        "name": ["phenol", "ethanol"],
        "formula": ["C6H6O", "C2H6O"],
        "monoisotopic_mass": [94.041865, 46.041865],
    }
    if extra_column:
        data["builder_note"] = ["fixture-row-1", "fixture-row-2"]
    return pl.DataFrame(data)


def spectra_frame() -> pl.DataFrame:
    """Three spectra whose keys are the content hash of their stored values."""
    contexts = [
        dict(precursor_mz=None, precursor_type=None, collision_energy=None, ionization_mode="EI"),
        dict(precursor_mz=250.5, precursor_type="[M+H]+", collision_energy=30.0, ionization_mode="positive"),
        dict(precursor_mz=250.5, precursor_type="[M+H]+", collision_energy=30.0, ionization_mode="positive"),
    ]
    keys = [
        spectrum_content_hash(mz, intensity, **context)
        for (mz, intensity), context in zip(PEAK_SETS, contexts)
    ]
    return pl.DataFrame({
        "spectrum_key": keys,
        "hash_mode": ["content_hash"] * len(PEAK_SETS),
        "peaks_mz": [mz for mz, _ in PEAK_SETS],
        "peaks_intensity": [intensity for _, intensity in PEAK_SETS],
        "intensity_normalized": [False, False, False],
        "ms_level": [1, 2, 2],
        "precursor_mz": [context["precursor_mz"] for context in contexts],
        "precursor_type": [context["precursor_type"] for context in contexts],
        "collision_energy": [context["collision_energy"] for context in contexts],
        "ionization_mode": [context["ionization_mode"] for context in contexts],
    })



def linker_frame(spectra: pl.DataFrame, molecules: pl.DataFrame) -> pl.DataFrame:
    """n:m fixture: spec0 -> mol0 and mol1 (one spectrum, two molecules), etc."""
    if spectra.height == 0 or molecules.height == 0:
        return pl.DataFrame(schema=LINKER_SCHEMA)
    return pl.DataFrame({
        "spectrum_key": [
            spectra["spectrum_key"][0],
            spectra["spectrum_key"][0],
            spectra["spectrum_key"][1],
            spectra["spectrum_key"][2],
        ],
        "molecule_key": [
            molecules["molecule_key"][0],
            molecules["molecule_key"][1],
            molecules["molecule_key"][1],
            molecules["molecule_key"][0],
        ],
        "source_library": ["fixture-lib", "fixture-lib", "other-lib", "fixture-lib"],
        "source_file": ["a.msp", "b.msp", "b.msp", "c.msp"],
        "acquisition_context": [None, json.dumps({"instrument": "qtof"}), None, None],
    })


def big_molecules_frame(n: int) -> pl.DataFrame:
    """``n`` distinct alkane molecules with genuine ``smiles_hash`` keys."""
    smiles = ["C" * (i + 1) for i in range(n)]
    return pl.DataFrame({
        "molecule_key": [smiles_hash(s) for s in smiles],
        "key_mode": ["smiles_hash"] * n,
        "inchikey": [None] * n,
        "inchikey_skeleton": [None] * n,
        "smiles": smiles,
        "smiles_hash": [smiles_hash(s) for s in smiles],
        "inchi": [None] * n,
        "name": [None] * n,
        "formula": [None] * n,
        "monoisotopic_mass": [None] * n,
    })


@dataclass
class ContainerBundle:
    """A container on disk plus the frames it was written from."""

    root: Path
    meta: ContainerMeta
    molecules: pl.DataFrame
    spectra: pl.DataFrame
    linker: pl.DataFrame

    @property
    def container(self) -> DataSpecContainer:
        return DataSpecContainer(self.root)


def container_mode(values: list[str], *, default: str) -> str:
    unique = set(values)
    if not unique:
        return default
    return unique.pop() if len(unique) == 1 else "mixed"


# ---------------------------------------------------------------------------
# Mutation helpers (used by conformance tests to break exactly one thing)
# ---------------------------------------------------------------------------


def rewrite_table(root: Path, table: str, frame: pl.DataFrame) -> None:
    """Overwrite one table file, bypassing the writer's own checks."""
    frame.write_parquet(root / f"{table}.parquet")


def read_envelope_json(root: Path) -> dict:
    return json.loads((root / "container_meta.json").read_text(encoding="utf-8"))


def patch_envelope(root: Path, *, recompute_fingerprint: bool = True, **changes) -> dict:
    """Patch envelope fields on disk; recompute the fingerprint unless told not to."""
    from unimol_lite.dataspec import container_fingerprint

    data = read_envelope_json(root)
    data.update(changes)
    if recompute_fingerprint:
        data["fingerprint"] = container_fingerprint(ContainerMeta.from_dict(data))
    (root / "container_meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def overwrite_envelope(root: Path, **changes) -> dict:
    """Overwrite envelope fields verbatim (no validation, no fingerprint update)."""
    data = read_envelope_json(root)
    data.update(changes)
    (root / "container_meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def write_small_type_table(path: Path, frame: pl.DataFrame) -> None:
    """Write a table with pyarrow *small* types (``string`` / ``list``).

    The protocol requires readers to accept both the small and the large Arrow
    variants of the logical schema.
    """
    arrow = frame.to_arrow()
    fields = []
    for field in arrow.schema:
        if pa.types.is_large_string(field.type):
            fields.append(pa.field(field.name, pa.string()))
        elif pa.types.is_large_list(field.type):
            fields.append(pa.field(field.name, pa.list_(field.type.value_type)))
        else:
            fields.append(field)
    pq.write_table(arrow.cast(pa.schema(fields)), path)


__all__ = [
    "ContainerBundle",
    "INCHIKEY_ETHANOL",
    "INCHIKEY_PHENOL",
    "PEAK_SETS",
    "SCHEMA_VERSION",
    "SPEC_VERSION",
    "big_molecules_frame",
    "container_mode",
    "empty_table",
    "linker_frame",
    "molecules_frame",
    "overwrite_envelope",
    "patch_envelope",
    "read_envelope_json",
    "rewrite_table",
    "spectra_frame",
    "write_small_type_table",
]
