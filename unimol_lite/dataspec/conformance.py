"""Protocol conformance validator for containers.

``validate_container(root)`` returns a report of *errors* (protocol violations —
the container must be refused) and *warnings* (runnable but should be fixed).
It is the local gate for containers this adapter did not write, and the evidence
producer for the protocol's consistency contracts.

Validation coverage: envelope completeness / fingerprint / counts / declared hash
modes / shuffle strategy -> table files, contract columns and dtypes -> row-level
keys (uniqueness, key/hash mode consistency, hash recomputation) -> linker
referential integrity -> ``unimol_*`` extension payloads (when present).

Two honest limitations of this implementation, both reported as warnings rather
than silently passing:

- ``hash_mode="splash"`` spectrum keys are **not** recomputed (SPLASH is not
  implemented in this library) — they are flagged ``spectra.key_unverified``;
- ``inchi`` (full text) missing on a molecule row is a warning: the key space is
  still unambiguous, but an ambiguity cannot be adjudicated from the container.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl

from .envelope import (
    ENVELOPE_REQUIRED_KEYS,
    ContainerMeta,
    container_fingerprint,
    derive_mode,
)
from .hashes import inchikey_skeleton, smiles_hash, spectrum_content_hash
from .schema import (
    CONTAINER_META_FILENAME,
    HASH_MODE_SPLASH,
    KEY_MODE_INCHIKEY,
    KINDS,
    MOLECULE_KEY_MODES,
    MOLECULE_KEY_MODES_CONTAINER,
    REQUIRED_ROW_COLUMNS,
    SHUFFLE_STRATEGIES,
    SPEC_VERSION,
    SPECTRA_HASH_MODES,
    SPECTRA_HASH_MODES_CONTAINER,
    TABLE_LINKER,
    TABLE_MOLECULES,
    TABLE_NAMES,
    TABLE_SCHEMAS,
    TABLE_SPECTRA,
    UNIMOL_ATOMS_COLUMN,
    UNIMOL_COORDS_COLUMN,
    UNIMOL_KIND_COLUMN,
)

__all__ = ["Severity", "ConformanceIssue", "ConformanceReport", "validate_container"]

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ConformanceIssue:
    """One protocol violation or advisory."""

    code: str
    message: str
    table: Optional[str] = None
    severity: Severity = "error"


@dataclass
class ConformanceReport:
    """Validation result. ``ok`` means "no error-level issue"."""

    issues: list[ConformanceIssue] = field(default_factory=list)
    root: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> list[ConformanceIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[ConformanceIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"{head}: {len(self.errors())} error(s), {len(self.warnings())} warning(s) @ {self.root}"]
        lines += [f"  [{issue.severity}] {issue.code} ({issue.table}): {issue.message}" for issue in self.issues]
        return "\n".join(lines)


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ConformanceIssue] = []

    def error(self, code: str, table: Optional[str], message: str) -> None:
        self.issues.append(ConformanceIssue(code=code, message=message, table=table, severity="error"))

    def warning(self, code: str, table: Optional[str], message: str) -> None:
        self.issues.append(ConformanceIssue(code=code, message=message, table=table, severity="warning"))

    def report(self, root: Path) -> ConformanceReport:
        return ConformanceReport(issues=self.issues, root=str(root))


def validate_container(root: str | Path) -> ConformanceReport:
    """Validate a container against the protocol; returns a structured report."""
    root = Path(root)
    collector = _Collector()

    meta = _validate_envelope(root, collector)
    if meta is None:
        return collector.report(root)  # envelope untrustworthy: nothing else is meaningful

    actual_counts: dict[str, int] = {}
    frames: dict[str, pl.DataFrame] = {}
    missing_columns: dict[str, set[str]] = {}
    for table in TABLE_NAMES:
        path = root / f"{table}.parquet"
        if not path.is_file():
            collector.error("table.missing", table, f"表文件缺失: {path}")
            continue
        missing_columns[table] = _validate_table_schema(path, table, collector)
        frame = pl.read_parquet(path)
        frames[table] = frame
        actual_counts[table] = frame.height

    if meta.counts is not None:
        for table in TABLE_NAMES:
            actual = actual_counts.get(table)
            declared = getattr(meta.counts, table, None)
            if actual is not None and declared != actual:
                collector.error(
                    "envelope.counts_mismatch", table, f"counts.{table} 声明 {declared} != 实际 {actual}"
                )

    _validate_molecules(frames.get(TABLE_MOLECULES), meta, collector, missing_columns.get(TABLE_MOLECULES, set()))
    _validate_spectra(frames.get(TABLE_SPECTRA), meta, collector, missing_columns.get(TABLE_SPECTRA, set()))
    _validate_linker(frames.get(TABLE_LINKER), frames, collector, missing_columns)
    return collector.report(root)


def _validate_envelope(root: Path, collector: _Collector) -> Optional[ContainerMeta]:
    path = root / CONTAINER_META_FILENAME
    if not path.is_file():
        collector.error("envelope.missing", None, f"envelope 缺失: {path}")
        return None
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        collector.error("envelope.unparseable", None, f"envelope 不可解析: {exc}")
        return None
    if not isinstance(data, dict):
        collector.error("envelope.invalid", None, "envelope 必须是 JSON 对象")
        return None
    missing = [key for key in ENVELOPE_REQUIRED_KEYS if data.get(key) is None]
    if missing:
        collector.error("envelope.missing_field", None, f"envelope 缺少必备字段: {missing}")
        return None
    try:
        meta = ContainerMeta.from_dict(data)
    except Exception as exc:  # malformed sub-objects
        collector.error("envelope.invalid", None, f"envelope 校验失败: {exc}")
        return None
    if meta.spec_version.split(".")[0] != SPEC_VERSION.split(".")[0]:
        collector.error(
            "envelope.unknown_spec_version",
            None,
            f"spec 大版本不认识: {meta.spec_version}（本适配器支持 {SPEC_VERSION}）——应拒绝",
        )
    if meta.hash_modes.molecules not in MOLECULE_KEY_MODES_CONTAINER:
        collector.error(
            "envelope.hash_modes_invalid", TABLE_MOLECULES, f"molecules 容器级模式非法: {meta.hash_modes.molecules!r}"
        )
    if meta.hash_modes.spectra not in SPECTRA_HASH_MODES_CONTAINER:
        collector.error(
            "envelope.hash_modes_invalid", TABLE_SPECTRA, f"spectra 容器级模式非法: {meta.hash_modes.spectra!r}"
        )
    if meta.shuffle_strategy is not None and meta.shuffle_strategy not in SHUFFLE_STRATEGIES:
        collector.error("envelope.shuffle_strategy_invalid", None, f"shuffle_strategy 非法: {meta.shuffle_strategy!r}")
    recomputed = container_fingerprint(meta)
    if meta.fingerprint != recomputed:
        collector.error(
            "envelope.fingerprint_mismatch", None, f"契约指纹不自洽: 存储 {meta.fingerprint} != 重算 {recomputed}"
        )
    return meta


def _validate_table_schema(path: Path, table: str, collector: _Collector) -> set[str]:
    schema = TABLE_SCHEMAS[table]
    actual = pl.read_parquet_schema(path)
    missing: set[str] = set()
    for name, dtype in schema.items():
        if name not in actual:
            collector.error("table.column_missing", table, f"契约列缺失: {name}")
            missing.add(name)
        elif actual[name] != dtype:
            collector.error("table.column_type", table, f"契约列 {name} dtype {actual[name]} != 契约 {dtype}")
    return missing


def _require_columns(
    frame: pl.DataFrame,
    table: str,
    needed: tuple[str, ...],
    collector: _Collector,
    already_reported: set[str],
) -> bool:
    """Guard row-level checks: a missing column is reported once, then skipped."""
    missing = [name for name in needed if name not in frame.columns and name not in already_reported]
    for name in missing:
        collector.error("table.column_missing", table, f"契约列缺失: {name}（行级校验跳过）")
    return all(name in frame.columns for name in needed)


def _validate_molecules(
    frame: Optional[pl.DataFrame],
    meta: ContainerMeta,
    collector: _Collector,
    already_reported: set[str],
) -> None:
    if frame is None or frame.height == 0:
        return
    if not _require_columns(
        frame, TABLE_MOLECULES, REQUIRED_ROW_COLUMNS[TABLE_MOLECULES], collector, already_reported
    ):
        return

    legal_modes = {mode for mode in frame["key_mode"].drop_nulls().to_list() if mode in MOLECULE_KEY_MODES}
    actual = derive_mode(legal_modes, legal=MOLECULE_KEY_MODES, default=meta.hash_modes.molecules)
    if actual != meta.hash_modes.molecules:
        collector.error(
            "envelope.hash_modes_mismatch",
            TABLE_MOLECULES,
            f"envelope 声明 molecules={meta.hash_modes.molecules!r}，行级实际 {sorted(legal_modes)}",
        )

    keys = frame["molecule_key"]
    if keys.null_count():
        collector.error("molecules.key_null", TABLE_MOLECULES, f"molecule_key 空值 {keys.null_count()} 行")
    duplicated = frame.filter(pl.col("molecule_key").is_not_null() & pl.col("molecule_key").is_duplicated()).height
    if duplicated:
        collector.error("molecules.key_dup", TABLE_MOLECULES, f"molecule_key 重复 {duplicated} 行")

    inchi_missing = 0
    for row in frame.iter_rows(named=True):
        key = row["molecule_key"]
        mode = row["key_mode"]
        if mode not in MOLECULE_KEY_MODES:
            collector.error("molecules.key_mode_invalid", TABLE_MOLECULES, f"key_mode 非法: {mode!r}（{key!r}）")
            continue
        if not row["smiles"]:
            collector.error("molecules.smiles_null", TABLE_MOLECULES, f"smiles 必填（{key!r}）")
        if not row["inchi"]:
            inchi_missing += 1
        if mode == KEY_MODE_INCHIKEY:
            if row["inchikey"] is None or row["inchikey"] != key:
                collector.error(
                    "molecules.key_mode_inconsistent",
                    TABLE_MOLECULES,
                    f"key_mode=inchikey 但 inchikey 空/与键不一致（{key!r}）",
                )
            if row["smiles_hash"] is not None:
                collector.error(
                    "molecules.key_mode_inconsistent",
                    TABLE_MOLECULES,
                    f"key_mode=inchikey 但 smiles_hash 非空（{key!r}）",
                )
        else:
            if row["smiles_hash"] is None or row["smiles_hash"] != key:
                collector.error(
                    "molecules.key_mode_inconsistent",
                    TABLE_MOLECULES,
                    f"key_mode=smiles_hash 但 smiles_hash 空/与键不一致（{key!r}）",
                )
            elif row["smiles"] and smiles_hash(row["smiles"]) != row["smiles_hash"]:
                collector.error(
                    "molecules.smiles_hash_mismatch",
                    TABLE_MOLECULES,
                    f"smiles_hash 与 sha256(smiles) 不一致（{key!r}）",
                )
        if row["inchikey"] is not None:
            try:
                expected = inchikey_skeleton(row["inchikey"])
            except ValueError as exc:
                collector.error("molecules.skeleton_mismatch", TABLE_MOLECULES, f"inchikey 非法: {exc}（{key!r}）")
                continue
            if row["inchikey_skeleton"] != expected:
                collector.error(
                    "molecules.skeleton_mismatch",
                    TABLE_MOLECULES,
                    f"inchikey_skeleton 与 inchikey 前 25 字符不一致（{key!r}）",
                )
    if inchi_missing:
        collector.warning(
            "molecules.inchi_missing",
            TABLE_MOLECULES,
            f"{inchi_missing} 行缺 inchi 全文——键歧义时无原文可裁决",
        )
    _validate_unimol_extension(frame, collector)


def _validate_unimol_extension(frame: pl.DataFrame, collector: _Collector) -> None:
    """Check the ``unimol_*`` payload columns when the container carries them."""
    if UNIMOL_KIND_COLUMN not in frame.columns:
        return
    for row in frame.iter_rows(named=True):
        kind = row[UNIMOL_KIND_COLUMN]
        key = row["molecule_key"]
        if kind is None:
            collector.error("extension.unimol_kind_invalid", TABLE_MOLECULES, f"unimol_kind 为空（{key!r}）")
            continue
        if kind not in KINDS:
            collector.error(
                "extension.unimol_kind_invalid", TABLE_MOLECULES, f"unimol_kind 非法: {kind!r}（可选 {list(KINDS)}）"
            )
            continue
        if kind != "pregenerated":
            continue
        atoms = row.get(UNIMOL_ATOMS_COLUMN)
        coords = row.get(UNIMOL_COORDS_COLUMN)
        if not atoms or coords is None:
            collector.error(
                "extension.unimol_payload_invalid", TABLE_MOLECULES, f"pregenerated 记录缺 atoms/coords（{key!r}）"
            )
            continue
        if len(coords) != 3 * len(atoms):
            collector.error(
                "extension.unimol_payload_invalid",
                TABLE_MOLECULES,
                f"pregenerated 记录坐标长度 {len(coords)} != 3 x 原子数 {len(atoms)}（{key!r}）",
            )


def _validate_spectra(
    frame: Optional[pl.DataFrame],
    meta: ContainerMeta,
    collector: _Collector,
    already_reported: set[str],
) -> None:
    if frame is None or frame.height == 0:
        return
    if not _require_columns(frame, TABLE_SPECTRA, REQUIRED_ROW_COLUMNS[TABLE_SPECTRA], collector, already_reported):
        return

    legal_modes = {mode for mode in frame["hash_mode"].drop_nulls().to_list() if mode in SPECTRA_HASH_MODES}
    actual = derive_mode(legal_modes, legal=SPECTRA_HASH_MODES, default=meta.hash_modes.spectra)
    if actual != meta.hash_modes.spectra:
        collector.error(
            "envelope.hash_modes_mismatch",
            TABLE_SPECTRA,
            f"envelope 声明 spectra={meta.hash_modes.spectra!r}，行级实际 {sorted(legal_modes)}",
        )

    keys = frame["spectrum_key"]
    if keys.null_count():
        collector.error("spectra.key_null", TABLE_SPECTRA, f"spectrum_key 空值 {keys.null_count()} 行")
    duplicated = frame.filter(pl.col("spectrum_key").is_not_null() & pl.col("spectrum_key").is_duplicated()).height
    if duplicated:
        collector.error("spectra.key_dup", TABLE_SPECTRA, f"spectrum_key 重复 {duplicated} 行")
    if frame["intensity_normalized"].null_count():
        collector.error("spectra.intensity_normalized_null", TABLE_SPECTRA, "intensity_normalized 存在空值")

    unverified = 0
    for row in frame.iter_rows(named=True):
        key = row["spectrum_key"]
        mode = row["hash_mode"]
        if mode not in SPECTRA_HASH_MODES:
            collector.error("spectra.hash_mode_invalid", TABLE_SPECTRA, f"hash_mode 非法: {mode!r}（{key!r}）")
        mz, intensity = row["peaks_mz"], row["peaks_intensity"]
        if mz is None or intensity is None or not len(mz) or len(mz) != len(intensity):
            collector.error("spectra.peaks_invalid", TABLE_SPECTRA, f"峰列表非法（空/长度不等）（{key!r}）")
            continue
        if max(intensity) <= 0:
            collector.error("spectra.peaks_invalid", TABLE_SPECTRA, f"峰列表非法（最大强度 ≤ 0）（{key!r}）")
            continue
        if not (all(math.isfinite(v) for v in mz) and all(math.isfinite(v) for v in intensity)):
            # P3-a：NaN/Inf 峰对 SPLASH 与 content_hash 都不可哈希——splash 分支此前
            # 直接透传逃逸，现与 content_hash 分支同口径报违规（对齐金标准行为）。
            collector.error("spectra.peaks_invalid", TABLE_SPECTRA, f"峰列表含 NaN/Inf（{key!r}）")
            continue
        if mode == HASH_MODE_SPLASH:
            unverified += 1
            continue
        try:
            recomputed = spectrum_content_hash(
                mz,
                intensity,
                precursor_mz=row["precursor_mz"],
                precursor_type=row["precursor_type"],
                ionization_mode=row["ionization_mode"],
                collision_energy=row["collision_energy"],
            )
        except (ValueError, OverflowError) as exc:
            collector.error("spectra.peaks_invalid", TABLE_SPECTRA, f"哈希重算失败（峰不可哈希）: {exc}（{key!r}）")
            continue
        if key != recomputed:
            collector.error(
                "spectra.key_mismatch",
                TABLE_SPECTRA,
                f"spectrum_key 与存储峰值的 content_hash 不一致（存储 {key!r} != 重算 {recomputed!r}）",
            )
    if unverified:
        collector.warning(
            "spectra.key_unverified",
            TABLE_SPECTRA,
            f"{unverified} 行 splash 键未重算（本适配器不实现 SPLASH，键按不透明标识透传）",
        )


def _fk_keys(frames: dict[str, pl.DataFrame], table: str, column: str) -> Optional[pl.DataFrame]:
    frame = frames.get(table)
    if frame is None or column not in frame.columns or frame.schema[column] != pl.String:
        return None  # missing/mistyped column already reported by the schema pass
    return frame[column].unique().to_frame()


def _validate_linker(
    frame: Optional[pl.DataFrame],
    frames: dict[str, pl.DataFrame],
    collector: _Collector,
    missing_columns: dict[str, set[str]],
) -> None:
    if frame is None or frame.height == 0:
        return
    if not _require_columns(
        frame, TABLE_LINKER, REQUIRED_ROW_COLUMNS[TABLE_LINKER], collector, missing_columns.get(TABLE_LINKER, set())
    ):
        return
    if any(frame.schema[column] != pl.String for column in ("spectrum_key", "molecule_key")):
        return  # dtype mismatch already reported by the schema pass
    if frame["spectrum_key"].null_count() or frame["molecule_key"].null_count():
        collector.error("linker.key_null", TABLE_LINKER, "linker 键列存在空值")
    spectra_keys = _fk_keys(frames, TABLE_SPECTRA, "spectrum_key")
    if spectra_keys is not None:
        dangling = frame.join(spectra_keys, on="spectrum_key", how="anti").height
        if dangling:
            collector.error("linker.fk_spectra_missing", TABLE_LINKER, f"spectrum_key 悬挂引用 {dangling} 行")
    molecule_keys = _fk_keys(frames, TABLE_MOLECULES, "molecule_key")
    if molecule_keys is not None:
        dangling = frame.join(molecule_keys, on="molecule_key", how="anti").height
        if dangling:
            collector.error("linker.fk_molecules_missing", TABLE_LINKER, f"molecule_key 悬挂引用 {dangling} 行")
