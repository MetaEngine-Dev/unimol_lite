"""Key and hash rules of the protocol, molecule side plus content-hash.

Implemented here:

- ``smiles_hash`` — SHA-256 of the canonical SMILES bytes (molecule fallback key);
- ``inchikey_skeleton`` — first 25 characters of an InChIKey (connectivity +
  stereo/isotope block, protonation block dropped);
- ``spectrum_content_hash`` — the protocol's guaranteed spectrum hash, computed
  over the stored peak values exactly as they are stored.

**SPLASH is intentionally not implemented.**  This library is molecule-domain:
it does not produce spectra, so it never needs to *derive* a spectrum key.  A
container that carries ``hash_mode="splash"`` rows (produced by a spectrum-side
tool) is read with its spectrum keys as opaque identifiers, and the conformance
report marks those keys as unverified rather than recomputing them.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Optional

__all__ = [
    "smiles_hash",
    "inchikey_skeleton",
    "validate_peaks",
    "spectrum_content_hash",
]


def smiles_hash(smiles: str) -> str:
    """SHA-256 hex digest of the canonical SMILES UTF-8 bytes.

    Canonicalisation is the *builder's* duty (see :mod:`unimol_lite.dataspec.records`,
    which canonicalises through RDKit before hashing).
    """
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("smiles 必须为非空字符串")
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def inchikey_skeleton(inchikey: str) -> str:
    """First 25 characters of an InChIKey (drops the protonation block)."""
    if not isinstance(inchikey, str) or len(inchikey) < 25:
        length = len(inchikey) if isinstance(inchikey, str) else "N/A"
        raise ValueError(f"inchikey 非法（长度 {length} < 25）")
    return inchikey[:25]


def validate_peaks(peaks_mz: Iterable[float], peaks_intensity: Iterable[float]) -> list[tuple[float, float]]:
    """Coerce and validate a peak list.

    Rejects degenerate spectra (empty peaks, mismatched lengths, non-finite
    values, maximum intensity ``<= 0``): the protocol forbids them in a
    container.
    """
    mz = [float(value) for value in peaks_mz]
    intensity = [float(value) for value in peaks_intensity]
    if len(mz) != len(intensity):
        raise ValueError(f"peaks_mz 与 peaks_intensity 长度不一致: {len(mz)} != {len(intensity)}")
    if not mz:
        raise ValueError("退化谱图（空峰列表）不可哈希，禁止入容器")
    pairs = list(zip(mz, intensity))
    for index, (mz_value, intensity_value) in enumerate(pairs):
        if not (math.isfinite(mz_value) and math.isfinite(intensity_value)):
            raise ValueError(
                f"峰含非有限值（Inf/NaN）不可哈希，禁止入容器: 第 {index} 峰 "
                f"mz={mz_value!r} intensity={intensity_value!r}"
            )
    if max(value for _, value in pairs) <= 0:
        raise ValueError("退化谱图（最大强度 ≤ 0）不可哈希，禁止入容器")
    return pairs


def _fmt(value: Optional[float], digits: int) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def spectrum_content_hash(
    peaks_mz: Iterable[float],
    peaks_intensity: Iterable[float],
    *,
    precursor_mz: Optional[float] = None,
    precursor_type: Optional[str] = None,
    ionization_mode: Optional[str] = None,
    collision_energy: Optional[float] = None,
) -> str:
    """Protocol content hash of one spectrum (64 hex characters).

    Peaks are sorted by ``(m/z ascending, intensity descending)`` and rendered as
    ``f"{mz:.4f}:{intensity:.2f}"`` joined by ``"|"``; the four context fields
    follow, missing ones as empty strings (``precursor_mz`` at 4 decimals,
    ``collision_energy`` at 2).  Values are used exactly as stored — this
    function never re-normalises intensities.
    """
    pairs = validate_peaks(peaks_mz, peaks_intensity)
    pairs.sort(key=lambda pair: (pair[0], -pair[1]))
    peak_string = "|".join(f"{mz:.4f}:{intensity:.2f}" for mz, intensity in pairs)
    fields = [
        peak_string,
        _fmt(precursor_mz, 4),
        precursor_type or "",
        ionization_mode or "",
        _fmt(collision_energy, 2),
    ]
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()
