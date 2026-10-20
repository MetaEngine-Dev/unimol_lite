import time
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import squareform
from torch.nn.functional import pdist
from .smiles2coords import (
    FeaturizationTimeout,
    OnTimeout,
    smiles2coords,
    _resolve_coord_mode,
    _warn_use_3d_deprecated,
)
from .utils import make_rng, _UNSET
from ..model.tokenizer import UniMolTokenizer

def generate_unimolitem(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    tokenizer: UniMolTokenizer = UniMolTokenizer(),
    random_seed: int = 42,
    max_atoms: int = 256,
    normalize_coord: bool = True,
    *addl_seeds
) -> dict[str, torch.Tensor]:
    """Generate input dictionary for UniMol model.

    Args:
        atoms (np.ndarray): np.ndarray of atom symbols, shape (n_atoms,)
        coordinates (np.ndarray): np.ndarray of coordinates, shape (n_atoms, 3)
        tokenizer (UniMolTokenizer, optional): tokenizer object. Defaults to UniMolTokenizer().
        random_seed (int, optional): random seed effecting the random selection of atoms when atoms exceed max_atoms. Defaults to 42.
        max_atoms (int, optional): maximum number of atoms to consider. Defaults to 256.
        normalize_coord (bool, optional): whether to normalize the coordinates. Defaults to True.

    Returns:
        dict: dictionary containing the following keys:
            - src_tokens: tokenized atoms with cls and sep tokens
            - src_distance: distance matrix of atoms
            - src_coord: coordinates of atoms
            - src_edge_type: edge types of atoms
    """    
    assert len(atoms) == len(coordinates), "Atoms and coordinates shape mismatch"
    mask_hydrogen = atoms != "H"
    atoms = atoms[mask_hydrogen]
    coordinates = coordinates[mask_hydrogen]
    if max_atoms and len(atoms) > max_atoms:
        # local RNG: never touches the global numpy state (thread-safe)
        rng = make_rng(random_seed, *addl_seeds)
        index = rng.choice(len(atoms), max_atoms, replace=False)
        atoms = np.array(atoms)[index]
        coordinates = coordinates[index]
    if normalize_coord:
        coordinates = coordinates - coordinates.mean(axis=0)
    src_tokens = torch.tensor(tokenizer.encode(atoms.tolist()), dtype=torch.long)
    src_tokens = torch.cat(
        [
            torch.tensor([tokenizer.cls_token_id]), 
            src_tokens,
            torch.tensor([tokenizer.sep_token_id])
        ], 
        dim=0
    )
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates = torch.cat(
        [
            torch.full_like(coordinates[0], 0.0).unsqueeze(0), 
            coordinates,
            torch.full_like(coordinates[0], 0.0).unsqueeze(0)
        ], 
        dim=0
    )
    num_types = len(tokenizer)
    node_input = src_tokens.clone()
    src_edges = node_input.view(-1, 1) * num_types + node_input.view(1, -1)
    distances = torch.from_numpy(squareform(pdist(coordinates.view(-1, 3))).astype(np.float32))
    return dict(
        src_tokens  = src_tokens,
        src_distance = distances,
        src_coord = coordinates,
        src_edge_type = src_edges,
    )

def smiles2unimolitem(
    smiles: str,
    tokenizer: UniMolTokenizer = UniMolTokenizer(),
    coord_pool_count: int = 1,
    random_seed: int = 42,
    max_atoms: int = 256,
    normalize_coord: bool = True,
    use_3d=_UNSET,
    *addl_seeds,
    coord_mode=_UNSET,
    rng_seed: int = None,
    return_meta: bool = False,
    timeout: Optional[float] = None,
    on_timeout: OnTimeout = "fallback_2d",
) -> dict[str, torch.Tensor]:
    """Generate input dictionary for UniMol model from smiles string.

    Args:
        smiles (str, optional): input smiles string.
        tokenizer (UniMolTokenizer, optional): tokenizer object. Defaults to UniMolTokenizer().
        coord_pool_count (int, optional): count of generated coordinates. Defaults to 1.
        random_seed (int, optional): random seed effecting coordinate selection in coord_pool, and random selection of atoms when atoms exceed max_atoms. Defaults to 42.
        max_atoms (int, optional): maximum number of atoms to consider. Defaults to 256.
        normalize_coord (bool, optional): whether to normalize the coordinates. Defaults to True.
        use_3d (bool, optional): deprecated, mapped to ``coord_mode``
            (``True`` -> ``"3d"``, ``False`` -> ``"auto"``).
        coord_mode (str, optional): ``"auto"`` (<=400 atoms -> 3D, >400 -> 2D),
            ``"3d"`` or ``"2d"``. Defaults to ``"auto"``.
        rng_seed (int, optional): seed for all random draws; defaults to ``random_seed``.
        return_meta (bool, optional): also return a meta dict. Defaults to False.
        timeout (float | None, optional): per-conformer wall-clock budget in
            seconds (embed + MMFF). ``None`` (default) keeps the historical
            untimed path. See :func:`smiles2coords`.
        on_timeout (str, optional): ``"fallback_2d"`` / ``"error"`` / ``"skip"``,
            forwarded to :func:`smiles2coords`. With ``"error"`` and
            ``return_meta=False`` a :class:`FeaturizationTimeout` is raised.

    Returns:
        dict: dictionary containing the following keys:
            - src_tokens: tokenized atoms with cls and sep tokens
            - src_distance: distance matrix of atoms
            - src_coord: coordinates of atoms
            - src_edge_type: edge types of atoms

        When ``return_meta=True`` returns ``(dict | None, meta)``; invalid SMILES
        yields ``(None, meta)`` with ``status="invalid"``.
    """
    start = time.perf_counter()
    seed = random_seed if rng_seed is None else rng_seed

    # resolve the deprecated arg here (warning points at this function's caller)
    # and hand ``smiles2coords`` a plain coord_mode, so it warns only once.
    if use_3d is not _UNSET or isinstance(coord_mode, bool):
        _warn_use_3d_deprecated()
    mode = _resolve_coord_mode(coord_mode, use_3d)

    # meta is always requested internally: ``status``/``reason`` decide both the
    # legacy return shapes and the exception below.
    coords_result, meta = smiles2coords(
        smiles,
        coord_pool_count,
        coord_mode=mode,
        return_meta=True,
        timeout=timeout,
        on_timeout=on_timeout,
    )
    if coords_result is None:
        if return_meta:
            meta["elapsed_s"] = time.perf_counter() - start
            return None, meta
        if meta["status"] == "timeout_error":
            raise FeaturizationTimeout(
                f"featurization of {smiles!r} exceeded timeout: {meta.get('reason')}"
            )
        if meta["status"] == "skipped":
            return None
        # legacy: an unparseable SMILES without meta raised TypeError
        raise TypeError(f"cannot featurize SMILES {smiles!r}: {meta['status']}")
    atoms, coordinates_list = coords_result

    rng = make_rng(seed, *addl_seeds)
    sample_idx = rng.randint(len(coordinates_list))
    coordinates = coordinates_list[sample_idx]
    item = generate_unimolitem(
        atoms = atoms,
        coordinates = coordinates,
        tokenizer = tokenizer,
        random_seed = seed,
        max_atoms = max_atoms,
        normalize_coord = normalize_coord,
        *addl_seeds
    )
    if return_meta:
        meta["elapsed_s"] = time.perf_counter() - start
        return item, meta
    return item
