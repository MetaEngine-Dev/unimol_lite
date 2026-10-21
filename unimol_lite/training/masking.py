"""Masked-atom pretraining transforms for Uni-Mol.

The original Uni-Mol pretraining code masks atom tokens, perturbs the coordinates
of the masked atoms, and computes three losses:

* masked token cross-entropy;
* masked coordinate Smooth-L1;
* masked pair-distance Smooth-L1.

This module implements the same data-side transform on the item format produced
by :func:`unimol_lite.data.smiles2unimolitem`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from ..model.tokenizer import UniMolTokenizer


def _rand(shape, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape)


def _noise_coords(
    coords: np.ndarray,
    mask: np.ndarray,
    noise_type: str,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_type == "none" or noise == 0:
        return coords.copy()
    selected = int(mask.sum())
    if selected == 0:
        return coords.copy()
    if noise_type == "uniform":
        delta = rng.uniform(-noise, noise, size=(selected, 3))
    elif noise_type == "normal":
        delta = rng.normal(0.0, noise, size=(selected, 3))
    elif noise_type == "trunc_normal":
        delta = np.clip(
            rng.normal(0.0, noise, size=(selected, 3)), -2 * noise, 2 * noise
        )
    else:
        raise ValueError("unknown noise_type: %r" % noise_type)
    result = coords.copy()
    result[mask] = result[mask] + delta
    return result


def mask_unimol_item(
    item: dict,
    tokenizer: UniMolTokenizer,
    *,
    mask_prob: float = 0.15,
    leave_unmasked_prob: float = 0.05,
    random_token_prob: float = 0.05,
    noise_type: str = "uniform",
    noise: float = 1.0,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Create a masked pretraining example from a featurized item.

    Returns a dictionary with the model inputs plus ``tokens_target``,
    ``coord_target`` and ``distance_target``.
    """
    if not item:
        raise ValueError("cannot mask an empty item")
    rng = np.random.default_rng(seed)
    tokens = item["src_tokens"].detach().cpu().numpy().astype(np.int64)
    coords = item["src_coord"].detach().cpu().numpy().astype(np.float32)
    length = tokens.shape[0]
    if length < 3:
        raise ValueError("sequence is too short to mask")
    atom_positions = np.arange(1, length - 1)
    selected = rng.random(atom_positions.shape[0]) < mask_prob
    if not selected.any():
        selected[rng.integers(0, selected.shape[0])] = True
    mask = np.zeros(length, dtype=bool)
    mask[atom_positions[selected]] = True

    tokens_target = tokens.copy()
    coord_target = coords.copy()
    noised_tokens = tokens.copy()
    mask_token_id = int(tokenizer.get_vocab()["[MASK]"])
    random_low, random_high = 4, len(tokenizer) - 1  # skip specials and [MASK]
    for position in np.flatnonzero(mask):
        draw = rng.random()
        if draw < random_token_prob:
            noised_tokens[position] = int(rng.integers(random_low, random_high))
        elif draw < random_token_prob + leave_unmasked_prob:
            noised_tokens[position] = tokens[position]
        else:
            noised_tokens[position] = mask_token_id

    noised_coords = _noise_coords(coords, mask, noise_type, noise, rng)

    # The encoder sees distances of the perturbed coordinates; the distance
    # head is trained against the original pairwise distances.
    src_coord = torch.from_numpy(noised_coords).float()
    coord_target = torch.from_numpy(coord_target).float()
    src_distance = torch.cdist(src_coord, src_coord).float()
    distance_target = torch.cdist(coord_target, coord_target).float()
    src_tokens_t = torch.from_numpy(noised_tokens).long()
    src_edge_type = (
        src_tokens_t.view(-1, 1) * len(tokenizer) + src_tokens_t.view(1, -1)
    ).long()
    return {
        "src_tokens": src_tokens_t,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
        "tokens_target": torch.from_numpy(tokens_target).long(),
        "coord_target": coord_target,
        "distance_target": distance_target,
    }
