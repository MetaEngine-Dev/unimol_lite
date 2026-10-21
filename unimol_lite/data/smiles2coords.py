import math
import numbers
import time
import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit import RDLogger

from .utils import _UNSET

RDLogger.DisableLog("rdApp.*")

COORD_MODES = ("auto", "3d", "2d")
ON_TIMEOUT_ACTIONS = ("fallback_2d", "error", "skip")

CoordMode = Literal["auto", "3d", "2d"]
OnTimeout = Literal["fallback_2d", "error", "skip"]

#: ``EmbedMolecule(mol, randomSeed=...)`` (the legacy overload) internally builds
#: an ``EmbedParameters`` object with these ETKDG flags **enabled**, while a bare
#: ``rdDistGeom.EmbedParameters()`` has them all **disabled**
#: (``useExpTorsionAnglePrefs=False`` / ``useBasicKnowledge=False`` /
#: ``useMacrocycleTorsions=False`` / ``useMacrocycle14config=False``).
#: They must be re-enabled explicitly, otherwise every conformer generated on
#: the ``timeout`` path would silently differ from the legacy path
#: (verified: ``timeout=None`` vs ``timeout=60`` bitwise identical only with
#: these flags set).
_LEGACY_ETKDG_FLAGS = dict(
    useExpTorsionAnglePrefs=True,
    useBasicKnowledge=True,
    useMacrocycleTorsions=True,
    useMacrocycle14config=True,
)


class FeaturizationTimeout(RuntimeError):
    """3D conformer generation exceeded the caller-provided ``timeout``.

    Raised only when ``on_timeout="error"`` and ``return_meta=False``; with
    ``return_meta=True`` the same condition yields ``(None, meta)`` with
    ``meta["status"] == "timeout_error"``.
    """


def mol2coords_2d(mol) -> np.ndarray:
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {Chem.MolToSmiles(mol)}"
    return coordinates


def _native_timeout_seconds(timeout: float) -> int:
    """RDKit's ``EmbedParameters.timeout`` is an ``unsigned int`` in seconds, so a
    fractional ``timeout`` cannot be stored as-is (assigning a float raises a
    Boost.Python ``ArgumentError``). Round **up** with a 1s floor: the native
    abort then never fires *before* the caller's budget; the wall-clock check
    after embed/MMFF enforces the exact (possibly sub-second) budget.
    """
    return max(1, int(math.ceil(timeout)))


def _timeout_failure_count(params) -> int:
    """Number of embed attempts that RDKit aborted with ``EXCEEDED_TIMEOUT``.

    ``EmbedParameters.GetFailureCounts()`` returns a flat tuple of counts indexed
    by ``rdDistGeom.EmbedFailureCauses`` (only populated when
    ``params.trackFailures`` is True).
    """
    counts = params.GetFailureCounts()
    index = int(rdDistGeom.EmbedFailureCauses.EXCEEDED_TIMEOUT)
    if len(counts) <= index:
        return 0
    return int(counts[index])


def _embed_one(mol, seed: int, timeout: float):
    """Embed + MMFF-optimize one conformer within ``timeout``.

    Returns ``(positions, None)`` on success, ``(None, reason)`` otherwise where
    ``reason`` is ``"timeout"`` / ``"embed_failed"`` / ``"mmff_failed"``.

    A failure is **not** retried: same params + same seed is deterministic, so a
    retry only doubles the budget without any chance of a different outcome
    (this replaces the historical ``maxAttempts=5000`` untimed retry).
    """
    params = rdDistGeom.EmbedParameters()
    params.timeout = _native_timeout_seconds(timeout)
    # 0 == unlimited, exactly like the legacy ``EmbedMolecule(mol, randomSeed=...)``
    # overload; a finite cap made some legitimately-hard molecules fail on the
    # timeout path only (P1, verification in ``04-施工日志.md``).
    params.maxIterations = 0
    params.randomSeed = seed
    params.trackFailures = True  # required by GetFailureCounts()
    for name, value in _LEGACY_ETKDG_FLAGS.items():
        setattr(params, name, value)

    start = time.perf_counter()
    try:
        work = deepcopy(mol)
        res = AllChem.EmbedMolecule(work, params)
    except Exception:
        return None, "embed_failed"
    if res != 0:
        if _timeout_failure_count(params) > 0 or (time.perf_counter() - start) >= timeout:
            return None, "timeout"
        return None, "embed_failed"

    try:
        AllChem.MMFFOptimizeMolecule(work, maxIters=200)  # some conformer can not use MMFF optimize
    except Exception:
        return None, "mmff_failed"
    # MMFF is inside the same budget: a slow optimization degrades as well
    if (time.perf_counter() - start) >= timeout:
        return None, "timeout"
    return work.GetConformer().GetPositions(), None


def _mol2coords_3d_legacy(mol, count):
    """Historical (``timeout=None``) path: no ``EmbedParameters``, no budget."""
    coordinate_list = []
    for seed in range(count):
        try:
            res = AllChem.EmbedMolecule(
                mol, randomSeed=seed
            )  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res != 0:
                # the historical maxAttempts=5000 untimed retry is removed on purpose
                return None, "embed_failed"
            AllChem.MMFFOptimizeMolecule(mol)  # some conformer can not use MMFF optimize
            coordinates = mol.GetConformer().GetPositions()
        except Exception:
            return None, "embed_failed"
        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {Chem.MolToSmiles(mol)}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list, None


def _mol2coords_3d_with_timeout(mol, count, timeout):
    """``timeout > 0`` path: every conformer gets ``timeout``, total budget is
    ``count * timeout``. A single failing conformer discards **all** 3D results
    so a molecule never comes back with a half-degraded conformer set."""
    deadline = time.perf_counter() + count * timeout
    coordinate_list = []
    for seed in range(count):
        positions, reason = _embed_one(mol, seed, timeout)
        # the total budget is checked after *every* attempt, including the
        # failing one (a single conformer can already blow a tiny budget)
        if time.perf_counter() > deadline:
            return None, "timeout"
        if positions is None:
            return None, reason
        coordinate_list.append(positions.astype(np.float32))
    return coordinate_list, None


def _mol2coords_3d(mol, count, timeout=None):
    """Return ``(coordinate_list, reason)``; ``coordinate_list`` is ``None`` on
    failure and ``reason`` then says why."""
    if timeout is None:
        return _mol2coords_3d_legacy(mol, count)
    return _mol2coords_3d_with_timeout(mol, count, timeout)


def _fallback_2d(mol, count) -> List[np.ndarray]:
    """2D fallback computed on a **clean copy** (no residual 3D conformer).

    Each entry is an independent copy: callers must not be able to mutate one
    conformer through another (``[arr] * (count + 1)`` aliases the same array).
    """
    coordinates = mol2coords_2d(deepcopy(mol))
    return [coordinates.copy() for _ in range(count + 1)]


def mol2coords_3d(mol, count, timeout: Optional[float] = None, on_timeout: OnTimeout = "fallback_2d"):
    """Generate ``count`` 3D conformers, applying ``on_timeout`` on failure.

    Args:
        mol: RDKit molecule (hydrogens already added).
        count (int): number of conformers.
        timeout (float | None): per-conformer budget in seconds; ``None`` keeps
            the historical untimed path.
        on_timeout (str): ``"fallback_2d"`` (2D coordinates, ``count + 1`` of
            them), ``"error"`` (raise :class:`FeaturizationTimeout`) or
            ``"skip"`` (return ``None``).

    Returns:
        list of ``(n_atoms, 3)`` float32 arrays, or ``None`` for ``"skip"``.
    """
    coordinate_list, reason = _mol2coords_3d(mol, count, timeout)
    if coordinate_list is not None:
        return coordinate_list
    if on_timeout == "fallback_2d":
        return _fallback_2d(mol, count)
    if on_timeout == "error":
        raise FeaturizationTimeout(f"3D conformer generation failed: {reason}")
    return None


def _warn_use_3d_deprecated():
    """Emit the ``use_3d`` deprecation warning pointing at the *caller* of the
    public API (helper -> public function -> caller)."""
    warnings.warn(
        "`use_3d` is deprecated, use `coord_mode` instead "
        '(`use_3d=True` -> `coord_mode="3d"`, `use_3d=False` -> `coord_mode="auto"`).',
        DeprecationWarning,
        stacklevel=3,
    )


def _validate_timeout(timeout):
    """``timeout`` must be ``None`` or a finite number ``> 0``."""
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
        raise ValueError(f"timeout must be None or a positive number, got {timeout!r}")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout must be None or > 0, got {timeout!r}")
    return value


def _validate_on_timeout(on_timeout):
    if on_timeout not in ON_TIMEOUT_ACTIONS:
        raise ValueError(f"on_timeout must be one of {ON_TIMEOUT_ACTIONS}, got {on_timeout!r}")
    return on_timeout


def _resolve_coord_mode(coord_mode, use_3d) -> str:
    """Resolve ``coord_mode`` / legacy ``use_3d`` into ``"auto" | "3d" | "2d"``.

    - ``coord_mode`` explicitly given -> it wins.
    - only legacy ``use_3d`` given -> ``True`` maps to ``"3d"``, ``False`` to
      ``"auto"`` (``"auto"`` is the historical ``use_3d=False`` inference path).
    """
    if isinstance(coord_mode, bool):  # legacy positional call: smiles2coords(smi, n, use_3d)
        use_3d = coord_mode
        coord_mode = _UNSET

    if use_3d is not _UNSET and coord_mode is _UNSET:
        return "3d" if use_3d else "auto"

    mode = "auto" if coord_mode is _UNSET else coord_mode
    if mode not in COORD_MODES:
        raise ValueError(f"coord_mode must be one of {COORD_MODES}, got {mode!r}")
    return mode


def smiles2coords(
    smiles: str,
    count: int = 10,
    coord_mode=_UNSET,
    return_meta: bool = False,
    use_3d=_UNSET,
    *,
    timeout: Optional[float] = None,
    on_timeout: OnTimeout = "fallback_2d",
) -> Tuple[np.ndarray, list]:
    """_summary_

    Args:
        smiles (str): smiles string
        count (int, optional): conformer num,all==11, 10 * 3d + 1 * 2d. Defaults to 10.
        coord_mode (str, optional): ``"auto"`` (<=400 atoms -> 3D, >400 -> 2D),
            ``"3d"`` (force 3D) or ``"2d"`` (force 2D, no conformer embedding).
            Defaults to ``"auto"``.
        return_meta (bool, optional): also return a meta dict. Defaults to False.
        use_3d (bool, optional): deprecated, mapped to ``coord_mode``
            (``True`` -> ``"3d"``, ``False`` -> ``"auto"``).
        timeout (float | None, optional): per-conformer wall-clock budget in
            seconds covering embed **and** MMFF. ``None`` (default) keeps the
            historical untimed path. For ``count > 1`` the total budget is
            ``count * timeout`` and any single failure discards all 3D results.
        on_timeout (str, optional): what to do when the budget is exceeded or
            embedding fails — ``"fallback_2d"`` (2D coordinates,
            ``status="degraded_2d"``), ``"error"`` (raise
            :class:`FeaturizationTimeout`, or ``status="timeout_error"`` with
            ``return_meta=True``) or ``"skip"`` (``None``,
            ``status="skipped"``).

    Returns:
        Tuple[np.ndarray, list]: atoms, coordinates list; or ``(result, meta)`` when
        ``return_meta=True``. Invalid SMILES returns ``(None, meta)`` when
        ``return_meta=True`` and the legacy ``[None, None]`` otherwise.
    """
    start = time.perf_counter()
    timeout = _validate_timeout(timeout)
    on_timeout = _validate_on_timeout(on_timeout)

    if use_3d is not _UNSET or isinstance(coord_mode, bool):
        _warn_use_3d_deprecated()
    mode = _resolve_coord_mode(coord_mode, use_3d)

    def _meta(status, used_mode, reason):
        return dict(
            status=status,
            coord_mode=used_mode,
            reason=reason,
            elapsed_s=time.perf_counter() - start,
        )

    # explicit forced mode is still reportable when the SMILES cannot be parsed
    forced_mode = mode if mode in ("3d", "2d") else None

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        if return_meta:
            return None, _meta("invalid", forced_mode, "parse_failed")
        return [None, None]

    try:
        mol = AllChem.AddHs(mol)
        num_atoms = len(mol.GetAtoms())
        if mode == "auto":
            used_mode = "2d" if num_atoms > 400 else "3d"
        else:
            used_mode = mode

        status, reason = "ok", None
        if used_mode == "2d":
            coordinate_list = [mol2coords_2d(mol)] * (count + 1)
        else:
            coordinate_list, reason = _mol2coords_3d(mol, count, timeout)
            if coordinate_list is None:
                if on_timeout == "error":
                    if return_meta:
                        return None, _meta("timeout_error", used_mode, reason)
                    raise FeaturizationTimeout(
                        f"3D conformer generation for {smiles!r} failed: {reason}"
                    )
                if on_timeout == "skip":
                    if return_meta:
                        return None, _meta("skipped", used_mode, reason)
                    return None
                # fallback_2d: degrade on a clean copy, report the produced mode
                coordinate_list = _fallback_2d(mol, count)
                status, used_mode = "degraded_2d", "2d"
            else:
                # add 2d conf
                coordinate_list.append(mol2coords_2d(mol).astype(np.float32))
        atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])  # after add H
    except FeaturizationTimeout:
        raise
    except Exception:
        if return_meta:
            return None, _meta("invalid", forced_mode, "failed")
        return [None, None]

    result = (atoms, coordinate_list)
    if return_meta:
        return result, _meta(status, used_mode, reason)
    return result