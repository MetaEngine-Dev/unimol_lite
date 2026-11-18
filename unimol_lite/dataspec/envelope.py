"""Container metadata envelope (``container_meta.json``).

The envelope carries the protocol version, provenance (``source`` / ``builder``),
row counts, declared hash modes, the declared shuffle strategy, the D5
placeholders (``metric`` / ``normalized``) and the **contract fingerprint**.

The fingerprint is the protocol's mixing guard: it is the SHA-256 of the
canonical JSON of ``{spec_version, schema_version, hash_modes, metric,
normalized}``.  Two containers whose fingerprints differ must not be mixed.

Deliberately dependency-free (dataclasses + ``json``): this library does not
depend on a validation framework.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .errors import DataSpecMismatchError, DataSpecValidationError
from .schema import (
    CONTAINER_META_FILENAME,
    MOLECULE_KEY_MODES_CONTAINER,
    SHUFFLE_STRATEGIES,
    SPEC_VERSION,
    SPECTRA_HASH_MODES_CONTAINER,
)

__all__ = [
    "SourceReference",
    "SourceInfo",
    "BuilderInfo",
    "CountsInfo",
    "HashModes",
    "ContainerMeta",
    "ENVELOPE_REQUIRED_KEYS",
    "container_fingerprint",
    "assert_compatible",
    "utc_now_iso",
    "derive_mode",
    "read_envelope",
    "write_envelope",
]

#: Keys that must be present (non-null) in a readable envelope.
ENVELOPE_REQUIRED_KEYS: tuple[str, ...] = (
    "spec_version",
    "schema_version",
    "source",
    "builder",
    "counts",
    "created_at",
    "hash_modes",
    "fingerprint",
)


@dataclass(frozen=True)
class SourceReference:
    """One provenance reference (source library, original file, digest)."""

    kind: str
    identifier: str
    digest: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "SourceReference":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"source.references 条目必须是对象, got {type(data).__name__}")
        kind = data.get("kind")
        identifier = data.get("identifier")
        if not isinstance(kind, str) or not kind:
            raise DataSpecValidationError("source.references 条目缺少 kind")
        if not isinstance(identifier, str) or not identifier:
            raise DataSpecValidationError("source.references 条目缺少 identifier")
        digest = data.get("digest")
        if digest is not None and not isinstance(digest, str):
            raise DataSpecValidationError("source.references.digest 必须是字符串或 null")
        return cls(kind=kind, identifier=identifier, digest=digest)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "identifier": self.identifier, "digest": self.digest}


@dataclass(frozen=True)
class SourceInfo:
    """Source-data digest plus a list of references."""

    digest: Optional[str] = None
    references: tuple[SourceReference, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Any) -> "SourceInfo":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"source 必须是对象, got {type(data).__name__}")
        digest = data.get("digest")
        if digest is not None and not isinstance(digest, str):
            raise DataSpecValidationError("source.digest 必须是字符串或 null")
        raw_refs = data.get("references") or []
        if not isinstance(raw_refs, (list, tuple)):
            raise DataSpecValidationError("source.references 必须是数组")
        return cls(digest=digest, references=tuple(SourceReference.from_dict(r) for r in raw_refs))

    def to_dict(self) -> dict[str, Any]:
        return {"digest": self.digest, "references": [r.to_dict() for r in self.references]}


@dataclass(frozen=True)
class BuilderInfo:
    """Builder identity; the place to self-declare canonicalisation choices."""

    name: str
    version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "BuilderInfo":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"builder 必须是对象, got {type(data).__name__}")
        name = data.get("name")
        version = data.get("version")
        if not isinstance(name, str) or not name:
            raise DataSpecValidationError("builder.name 必须是非空字符串")
        if version is not None and not isinstance(version, str):
            raise DataSpecValidationError("builder.version 必须是字符串或 null")
        return cls(name=name, version=version)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class CountsInfo:
    """Row counts of the three tables (reconciliation only; not fingerprinted)."""

    molecules: int
    spectra: int
    linker: int

    @classmethod
    def from_dict(cls, data: Any) -> "CountsInfo":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"counts 必须是对象, got {type(data).__name__}")
        values: dict[str, int] = {}
        for key in ("molecules", "spectra", "linker"):
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DataSpecValidationError(f"counts.{key} 必须是非负整数, got {value!r}")
            values[key] = value
        return cls(**values)

    def to_dict(self) -> dict[str, int]:
        return {"molecules": self.molecules, "spectra": self.spectra, "linker": self.linker}


@dataclass(frozen=True)
class HashModes:
    """Container-level hash-mode declaration (``mixed`` = both row modes used)."""

    spectra: str
    molecules: str

    @classmethod
    def from_dict(cls, data: Any) -> "HashModes":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"hash_modes 必须是对象, got {type(data).__name__}")
        spectra = data.get("spectra")
        molecules = data.get("molecules")
        if spectra not in SPECTRA_HASH_MODES_CONTAINER:
            raise DataSpecValidationError(
                f"hash_modes.spectra 非法: {spectra!r}（可选 {SPECTRA_HASH_MODES_CONTAINER}）"
            )
        if molecules not in MOLECULE_KEY_MODES_CONTAINER:
            raise DataSpecValidationError(
                f"hash_modes.molecules 非法: {molecules!r}（可选 {MOLECULE_KEY_MODES_CONTAINER}）"
            )
        return cls(spectra=str(spectra), molecules=str(molecules))

    def to_dict(self) -> dict[str, str]:
        return {"spectra": self.spectra, "molecules": self.molecules}


@dataclass(frozen=True)
class ContainerMeta:
    """The envelope. ``counts`` / ``created_at`` / ``fingerprint`` are filled in
    by the writer; readers require them to be present."""

    spec_version: str
    schema_version: str
    source: SourceInfo
    builder: BuilderInfo
    hash_modes: HashModes
    counts: Optional[CountsInfo] = None
    created_at: Optional[str] = None
    shuffle_strategy: Optional[str] = None
    metric: Optional[str] = None
    normalized: Optional[bool] = None
    fingerprint: Optional[str] = None

    # -- (de)serialisation ---------------------------------------------------
    @classmethod
    def from_dict(cls, data: Any) -> "ContainerMeta":
        if not isinstance(data, Mapping):
            raise DataSpecValidationError(f"envelope 必须是 JSON 对象, got {type(data).__name__}")
        for key in ("spec_version", "schema_version"):
            value = data.get(key)
            if not isinstance(value, str) or not value:
                raise DataSpecValidationError(f"envelope.{key} 必须是非空字符串")
        shuffle = data.get("shuffle_strategy")
        if shuffle is not None and shuffle not in SHUFFLE_STRATEGIES:
            raise DataSpecValidationError(
                f"shuffle_strategy 非法: {shuffle!r}（可选 {SHUFFLE_STRATEGIES} 或 null）"
            )
        metric = data.get("metric")
        if metric is not None and not isinstance(metric, str):
            raise DataSpecValidationError("envelope.metric 必须是字符串或 null")
        normalized = data.get("normalized")
        if normalized is not None and not isinstance(normalized, bool):
            raise DataSpecValidationError("envelope.normalized 必须是布尔或 null")
        created_at = data.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise DataSpecValidationError("envelope.created_at 必须是字符串或 null")
        counts_raw = data.get("counts")
        fingerprint = data.get("fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise DataSpecValidationError("envelope.fingerprint 必须是字符串或 null")
        return cls(
            spec_version=str(data["spec_version"]),
            schema_version=str(data["schema_version"]),
            source=SourceInfo.from_dict(data.get("source")),
            builder=BuilderInfo.from_dict(data.get("builder")),
            hash_modes=HashModes.from_dict(data.get("hash_modes")),
            counts=None if counts_raw is None else CountsInfo.from_dict(counts_raw),
            created_at=created_at,
            shuffle_strategy=shuffle,
            metric=metric,
            normalized=normalized,
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Envelope as a JSON-ready dict, in protocol field order."""
        return {
            "spec_version": self.spec_version,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "builder": self.builder.to_dict(),
            "counts": None if self.counts is None else self.counts.to_dict(),
            "created_at": self.created_at,
            "hash_modes": self.hash_modes.to_dict(),
            "shuffle_strategy": self.shuffle_strategy,
            "metric": self.metric,
            "normalized": self.normalized,
            "fingerprint": self.fingerprint,
        }

    def with_updates(self, **changes: Any) -> "ContainerMeta":
        """Return a copy with the given fields replaced."""
        values = {**self.__dict__, **changes}
        return ContainerMeta(**values)


def container_fingerprint(meta: ContainerMeta) -> str:
    """Contract fingerprint: SHA-256 over canonical JSON of the contract fields.

    Counting fields (``counts``) and provenance fields (``source``, ``builder``,
    ``created_at``, ``shuffle_strategy``) are excluded — rows may legally grow.
    """
    payload = {
        "spec_version": meta.spec_version,
        "schema_version": meta.schema_version,
        "hash_modes": {"spectra": meta.hash_modes.spectra, "molecules": meta.hash_modes.molecules},
        "metric": meta.metric,
        "normalized": meta.normalized,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_compatible(a: "ContainerMeta | str", b: "ContainerMeta | str") -> None:
    """Refuse to mix two containers whose contract fingerprints differ.

    Accepts ``ContainerMeta`` objects or ``"sha256:..."`` fingerprint strings, so
    either side (library build vs. query) can hold only the fingerprint.
    """
    fa = a if isinstance(a, str) else container_fingerprint(a)
    fb = b if isinstance(b, str) else container_fingerprint(b)
    if fa != fb:
        raise DataSpecMismatchError(f"容器契约指纹不一致，拒绝混用: {fa} != {fb}")


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_mode(mode_values: Iterable[Optional[str]], *, legal: tuple[str, ...], default: str) -> str:
    """Row-level mode set -> container-level declaration.

    One distinct legal value -> that value; two -> ``"mixed"``; none (empty table
    or all-null column) -> ``default`` (no row-level fact to reconcile against).
    Values outside ``legal`` are ignored here; they are reported as invalid by
    conformance validation.
    """
    valid = {value for value in mode_values if value in legal}
    if not valid:
        return default
    return valid.pop() if len(valid) == 1 else "mixed"


# ---------------------------------------------------------------------------
# File-level I/O
# ---------------------------------------------------------------------------


def read_envelope(root: str | Path) -> ContainerMeta:
    """Read and validate ``container_meta.json``.

    Fails closed: a missing file, unparsable JSON, missing required field, an
    unknown major protocol version or a fingerprint that does not match its own
    contract fields all raise :class:`DataSpecValidationError`.
    """
    path = Path(root) / CONTAINER_META_FILENAME
    if not path.is_file():
        raise DataSpecValidationError(f"envelope 缺失: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataSpecValidationError(f"envelope 不可解析: {path} ({exc})") from exc
    if not isinstance(data, Mapping):
        raise DataSpecValidationError(f"envelope 必须是 JSON 对象: {path}")
    missing = [key for key in ENVELOPE_REQUIRED_KEYS if data.get(key) is None]
    if missing:
        raise DataSpecValidationError(f"envelope 缺少必备字段: {missing}（{path}）")
    meta = ContainerMeta.from_dict(data)
    major, expected = meta.spec_version.split(".")[0], SPEC_VERSION.split(".")[0]
    if major != expected:
        raise DataSpecValidationError(
            f"spec 大版本不认识: {meta.spec_version}（本适配器实现 {SPEC_VERSION}）——拒绝读取"
        )
    recomputed = container_fingerprint(meta)
    if meta.fingerprint != recomputed:
        raise DataSpecValidationError(
            f"契约指纹不自洽: 存储 {meta.fingerprint} != 重算 {recomputed}"
        )
    return meta


def write_envelope(root: str | Path, meta: ContainerMeta) -> Path:
    """Write ``container_meta.json`` (caller is responsible for a finalised meta)."""
    path = Path(root) / CONTAINER_META_FILENAME
    path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
