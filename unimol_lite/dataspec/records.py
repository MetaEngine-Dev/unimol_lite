"""unimol_lite molecule records on top of the protocol container.

The container is molecule-domain here: a record is one molecule, and ``kind``
says how its payload is stored.  Payload lives in the ``unimol_*`` extension
columns (the protocol allows extra columns next to the contract columns).

Record kinds
------------

``smiles``
    Only the contract columns are meaningful; ``unimol_atoms`` /
    ``unimol_coords`` / ``unimol_coord_mode`` are null.  The molecule is
    featurized on read (RDKit conformer generation, so results depend on the
    reader's RDKit build and on ``coord_mode`` / ``rng_seed``).

``pregenerated``
    The contract columns plus a frozen featurization payload:
    ``unimol_atoms`` (atom symbols including hydrogens), ``unimol_coords``
    (flattened ``x, y, z, ...``, length ``3 * len(atoms)``) and
    ``unimol_coord_mode`` (``"3d"`` / ``"2d"``, the provenance of the stored
    conformer).  Reading rebuilds the model input from these coordinates, so no
    RDKit embedding happens — the tensors are reproducible across machines and
    across RDKit versions.

A container written by another tool has no ``unimol_*`` columns; its rows are
read as ``smiles`` records (canonical SMILES is a contract column).

Identity is the builder's duty: :func:`molecule_identity` canonicalises with
RDKit and keys the molecule by InChIKey, falling back to ``smiles_hash`` when
RDKit cannot produce one.  Both the key mode and the fallback are per-row flags,
so a container may mix the two.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from .container import DEFAULT_BLOCK_ROWS, DataSpecContainer, derive_hash_modes, write_container
from .envelope import BuilderInfo, ContainerMeta, SourceInfo
from .errors import DataSpecError, DataSpecValidationError
from .hashes import inchikey_skeleton, smiles_hash
from .schema import (
    KEY_MODE_INCHIKEY,
    KEY_MODE_SMILES_HASH,
    KIND_PREGENERATED,
    KIND_SMILES,
    KINDS,
    MOLECULE_RECORD_SCHEMA,
    SCHEMA_VERSION,
    SPEC_VERSION,
    TABLE_MOLECULES,
    UNIMOL_ATOMS_COLUMN,
    UNIMOL_COORD_MODE_COLUMN,
    UNIMOL_COORDS_COLUMN,
    UNIMOL_EXTENSION_COLUMNS,
    UNIMOL_KIND_COLUMN,
)

__all__ = [
    "KIND_SMILES",
    "KIND_PREGENERATED",
    "KINDS",
    "DEFAULT_RECORD_BATCH_ROWS",
    "default_builder",
    "molecule_identity",
    "build_record",
    "frame_from_records",
    "write_records",
    "iter_records",
    "record_to_item",
    "iter_items",
]

#: Rows per batch when streaming records out of a container.
DEFAULT_RECORD_BATCH_ROWS = 65_536

#: Contract columns of a molecule record, in write order.
_CONTRACT_COLUMNS: tuple[str, ...] = tuple(
    name for name in MOLECULE_RECORD_SCHEMA.names() if not name.startswith("unimol_")
)


def default_builder() -> BuilderInfo:
    """Builder identity stamped on containers written by this library."""
    from .. import __version__

    return BuilderInfo(name="unimol_lite", version=__version__)


# ---------------------------------------------------------------------------
# Building records
# ---------------------------------------------------------------------------


def molecule_identity(
    smiles: str,
    *,
    name: Optional[str] = None,
    canonicalize: bool = True,
    with_inchi: bool = True,
) -> dict[str, Any]:
    """Resolve the identity columns of one molecule.

    Canonicalisation and hashing are the builder's responsibility: SMILES is
    canonicalised with RDKit, the key is the InChIKey when RDKit can derive one
    (``key_mode="inchikey"``) and the SHA-256 of the canonical SMILES otherwise
    (``key_mode="smiles_hash"``).  ``inchi`` (full text) is kept when available
    so a key ambiguity can be adjudicated from the container.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"smiles 必须为非空字符串, got {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True) if canonicalize else smiles
    if not canonical:
        raise ValueError(f"无法生成 canonical SMILES: {smiles!r}")

    inchikey: Optional[str] = None
    try:
        inchikey = Chem.MolToInchiKey(mol) or None
    except Exception:  # noqa: BLE001 - RDKit raises RuntimeError on odd inputs
        inchikey = None

    inchi: Optional[str] = None
    if with_inchi:
        try:
            inchi = Chem.MolToInchi(mol) or None
        except Exception:  # noqa: BLE001
            inchi = None

    formula: Optional[str] = None
    mass: Optional[float] = None
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mass = float(Descriptors.ExactMolWt(mol))
    except Exception:  # noqa: BLE001 - descriptors are optional metadata
        pass

    canonical_hash = smiles_hash(canonical)
    if inchikey:
        key_mode = KEY_MODE_INCHIKEY
        key = inchikey
        skeleton: Optional[str] = inchikey_skeleton(inchikey)
        stored_smiles_hash: Optional[str] = None
    else:
        key_mode = KEY_MODE_SMILES_HASH
        key = canonical_hash
        skeleton = None
        stored_smiles_hash = canonical_hash

    return {
        "molecule_key": key,
        "key_mode": key_mode,
        "inchikey": inchikey,
        "inchikey_skeleton": skeleton,
        "smiles": canonical,
        "smiles_hash": stored_smiles_hash,
        "inchi": inchi,
        "name": name,
        "formula": formula,
        "monoisotopic_mass": mass,
    }


def build_record(
    smiles: str,
    *,
    kind: str = KIND_SMILES,
    name: Optional[str] = None,
    coord_mode: str = "auto",
    timeout: Optional[float] = 30.0,
    on_timeout: str = "fallback_2d",
    with_inchi: bool = True,
    canonicalize: bool = True,
) -> dict[str, Any]:
    """Build one molecule record of the requested ``kind``.

    For ``kind="pregenerated"`` the conformer is generated once here (one
    conformer, ``coord_mode`` / ``timeout`` / ``on_timeout`` honoured) and stored
    in the record; for ``kind="smiles"`` nothing beyond the identity columns is
    stored.

    Raises:
        ValueError: unknown ``kind``, unparsable SMILES, or a failed/unusable
            conformer generation for the pregenerated kind.
    """
    if kind not in KINDS:
        raise ValueError(f"未知记录类型: {kind!r}（可选 {list(KINDS)}）")
    record = molecule_identity(smiles, name=name, canonicalize=canonicalize, with_inchi=with_inchi)
    record[UNIMOL_KIND_COLUMN] = kind
    record[UNIMOL_ATOMS_COLUMN] = None
    record[UNIMOL_COORDS_COLUMN] = None
    record[UNIMOL_COORD_MODE_COLUMN] = None

    if kind == KIND_PREGENERATED:
        from ..data.smiles2coords import smiles2coords

        result, meta = smiles2coords(
            record["smiles"],
            count=1,
            coord_mode=coord_mode,
            return_meta=True,
            timeout=timeout,
            on_timeout=on_timeout,
        )
        if result is None:
            raise ValueError(
                f"无法为 {record['smiles']!r} 预生成构象: status={meta.get('status')} reason={meta.get('reason')}"
            )
        atoms, coordinate_list = result
        coordinates = np.asarray(coordinate_list[0], dtype=np.float64).reshape(-1, 3)
        record[UNIMOL_ATOMS_COLUMN] = [str(symbol) for symbol in atoms]
        record[UNIMOL_COORDS_COLUMN] = [float(value) for value in coordinates.reshape(-1)]
        record[UNIMOL_COORD_MODE_COLUMN] = meta.get("coord_mode")
    return record


def _normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing columns and validate the payload of one record."""
    row = {name: record.get(name) for name in _CONTRACT_COLUMNS}
    kind = record.get(UNIMOL_KIND_COLUMN) or KIND_SMILES
    if kind not in KINDS:
        raise DataSpecValidationError(f"未知记录类型: {kind!r}（可选 {list(KINDS)}）")
    row[UNIMOL_KIND_COLUMN] = kind
    for column in UNIMOL_EXTENSION_COLUMNS:
        if column != UNIMOL_KIND_COLUMN:
            row[column] = record.get(column)
    if not row["molecule_key"]:
        raise DataSpecValidationError("记录缺少 molecule_key（先用 molecule_identity/build_record 生成身份列）")
    if kind == KIND_PREGENERATED:
        atoms = row[UNIMOL_ATOMS_COLUMN]
        coords = row[UNIMOL_COORDS_COLUMN]
        if not atoms or coords is None:
            raise DataSpecValidationError(f"pregenerated 记录缺 atoms/coords（{row['molecule_key']!r}）")
        if len(coords) != 3 * len(atoms):
            raise DataSpecValidationError(
                f"pregenerated 记录坐标长度 {len(coords)} != 3 x 原子数 {len(atoms)}（{row['molecule_key']!r}）"
            )
    return row


def frame_from_records(records: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Records -> molecules frame (contract columns + ``unimol_*`` extension)."""
    rows = [_normalise_record(record) for record in records]
    if not rows:
        return pl.DataFrame(schema=MOLECULE_RECORD_SCHEMA)
    frame = pl.DataFrame(rows)
    return frame.cast({name: dtype for name, dtype in MOLECULE_RECORD_SCHEMA.items()}).select(
        MOLECULE_RECORD_SCHEMA.names()
    )


def write_records(
    root: str | Path,
    records: "pl.DataFrame | Iterable[Mapping[str, Any]]",
    *,
    source: Optional[SourceInfo] = None,
    builder: Optional[BuilderInfo] = None,
    shuffle_strategy: Optional[str] = None,
    metric: Optional[str] = None,
    normalized: Optional[bool] = None,
    spectra: Optional[pl.DataFrame] = None,
    linker: Optional[pl.DataFrame] = None,
    block_rows: int = DEFAULT_BLOCK_ROWS,
) -> ContainerMeta:
    """Write a molecule container from records and return the finalised envelope.

    ``shuffle_strategy`` is recorded as a declaration (protocol requirement for
    training use); ``metric`` / ``normalized`` are reserved placeholders that do
    participate in the contract fingerprint.

    ``spectra`` / ``linker`` are pass-through: supply frames only when the caller
    produced them elsewhere (this library neither derives nor verifies SPLASH
    keys).
    """
    frame = records if isinstance(records, pl.DataFrame) else frame_from_records(records)
    if not isinstance(frame, pl.DataFrame):
        raise DataSpecValidationError(f"records 必须是 DataFrame 或记录可迭代对象, got {type(frame).__name__}")
    if frame.height and "molecule_key" not in frame.columns:
        raise DataSpecValidationError("molecules 帧缺少 molecule_key 列")
    if frame.height:
        duplicated = frame.filter(pl.col("molecule_key").is_duplicated())
        if duplicated.height:
            sample = duplicated["molecule_key"].unique().head(5).to_list()
            raise DataSpecValidationError(
                f"molecule_key 必须容器内唯一（重复 {duplicated.height} 行，示例 {sample}）"
                "——同一分子的重复记录请先去重"
            )
    if frame.height and UNIMOL_KIND_COLUMN not in frame.columns:
        frame = frame.with_columns(pl.lit(KIND_SMILES).alias(UNIMOL_KIND_COLUMN))
    hash_modes = derive_hash_modes(frame, spectra)
    meta = ContainerMeta(
        spec_version=SPEC_VERSION,
        schema_version=SCHEMA_VERSION,
        source=source or SourceInfo(),
        builder=builder or default_builder(),
        hash_modes=hash_modes,
        shuffle_strategy=shuffle_strategy,
        metric=metric,
        normalized=normalized,
    )
    return write_container(
        root, frame, spectra=spectra, linker=linker, meta=meta, block_rows=block_rows
    )


# ---------------------------------------------------------------------------
# Reading records / model inputs
# ---------------------------------------------------------------------------


def iter_records(
    root: str | Path,
    *,
    kind: Optional[str] = None,
    batch_rows: int = DEFAULT_RECORD_BATCH_ROWS,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[dict[str, Any]]:
    """Stream molecule records out of a container, batch by batch.

    Rows of containers without ``unimol_*`` columns are reported as ``kind="smiles"``.
    ``columns`` narrows the projection (e.g. ``("smiles",)``) — contract columns
    are always readable, extension columns only when present.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"未知记录类型: {kind!r}（可选 {list(KINDS)}）")
    container = DataSpecContainer(root)
    # P3-b：kind 过滤依赖 unimol_kind 列——容器缺列时所有行都会默认成 smiles，
    # kind="pregenerated" 的过滤会**静默零行**。显式拒绝而不是静默空转。
    if kind is not None and kind != KIND_SMILES:
        schema_names = container.scan(TABLE_MOLECULES).collect_schema().names()
        if UNIMOL_KIND_COLUMN not in schema_names:
            raise ValueError(
                f"容器缺 {UNIMOL_KIND_COLUMN!r} 列，无法按 kind={kind!r} 过滤"
                "（过滤将静默零行）——请用 kind=None 读取或换带扩展列的容器"
            )
    for batch in container.iter_molecules_batches(batch_rows):
        if columns is not None:
            selected = [name for name in columns if name in batch.columns]
            missing = [name for name in columns if name not in batch.columns]
            if missing:
                raise DataSpecValidationError(f"容器缺列: {missing}")
            batch = batch.select(selected)
        has_kind = UNIMOL_KIND_COLUMN in batch.columns
        for row in batch.iter_rows(named=True):
            row_kind = (row.get(UNIMOL_KIND_COLUMN) if has_kind else None) or KIND_SMILES
            if kind is not None and row_kind != kind:
                continue
            row[UNIMOL_KIND_COLUMN] = row_kind
            yield row


def record_to_item(
    record: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    rng_seed: int = 42,
    max_atoms: int = 256,
    coord_mode: str = "auto",
    timeout: Optional[float] = 30.0,
    on_timeout: str = "fallback_2d",
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """One record -> ``(model input item | None, meta)``.

    ``pregenerated`` records rebuild the item from the stored coordinates (no
    RDKit embedding, reproducible); ``smiles`` records go through the standard
    featurization path, so ``coord_mode`` / ``timeout`` / ``rng_seed`` apply.
    """
    kind = record.get(UNIMOL_KIND_COLUMN) or KIND_SMILES
    key = record.get("molecule_key")
    start = time.perf_counter()

    if kind == KIND_PREGENERATED:
        from ..data.prepare_unimol_item import generate_unimolitem

        atoms = record.get(UNIMOL_ATOMS_COLUMN)
        coords = record.get(UNIMOL_COORDS_COLUMN)
        if not atoms or coords is None:
            return None, dict(
                status="invalid",
                reason="pregenerated_payload_missing",
                coord_mode=None,
                record_kind=kind,
                molecule_key=key,
                elapsed_s=time.perf_counter() - start,
            )
        if len(coords) != 3 * len(atoms):
            return None, dict(
                status="invalid",
                reason="pregenerated_payload_mismatch",
                coord_mode=None,
                record_kind=kind,
                molecule_key=key,
                elapsed_s=time.perf_counter() - start,
            )
        coordinates = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        kwargs: dict[str, Any] = dict(random_seed=rng_seed, max_atoms=max_atoms)
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        try:
            item = generate_unimolitem(np.asarray(atoms), coordinates, **kwargs)
        except Exception as exc:  # noqa: BLE001 - report per row, never abort a stream
            return None, dict(
                status="error",
                reason=type(exc).__name__,
                coord_mode=record.get(UNIMOL_COORD_MODE_COLUMN),
                record_kind=kind,
                molecule_key=key,
                elapsed_s=time.perf_counter() - start,
            )
        return item, dict(
            status="ok",
            reason=None,
            coord_mode=record.get(UNIMOL_COORD_MODE_COLUMN),
            record_kind=kind,
            molecule_key=key,
            elapsed_s=time.perf_counter() - start,
        )

    if kind == KIND_SMILES:
        from ..data.prepare_unimol_item import smiles2unimolitem

        smiles = record.get("smiles")
        if not smiles:
            return None, dict(
                status="invalid",
                reason="smiles_missing",
                coord_mode=None,
                record_kind=kind,
                molecule_key=key,
                elapsed_s=time.perf_counter() - start,
            )
        kwargs = dict(
            rng_seed=rng_seed,
            max_atoms=max_atoms,
            coord_mode=coord_mode,
            timeout=timeout,
            on_timeout=on_timeout,
            return_meta=True,
        )
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        item, meta = smiles2unimolitem(smiles, **kwargs)
        meta = dict(meta)
        meta["record_kind"] = kind
        meta["molecule_key"] = key
        return item, meta

    raise DataSpecError(f"未知记录类型: {kind!r}（可选 {list(KINDS)}）")


def iter_items(
    root: str | Path,
    *,
    kind: Optional[str] = None,
    batch_rows: int = DEFAULT_RECORD_BATCH_ROWS,
    **item_kwargs: Any,
) -> Iterator[tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any]]]:
    """Stream ``(record, item | None, meta)`` triples for a whole container.

    Failed rows are reported (``item is None``, ``meta["status"] != "ok"``) rather
    than raised, mirroring :func:`unimol_lite.featurize_batch` per-row isolation.
    """
    for record in iter_records(root, kind=kind, batch_rows=batch_rows):
        item, meta = record_to_item(record, **item_kwargs)
        yield record, item, meta
