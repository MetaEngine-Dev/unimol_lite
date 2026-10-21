"""Datasets and collation for Uni-Mol masked-atom pretraining."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lmdb
from torch import Tensor
from torch.utils.data import Dataset

from ..data.collate_fn import collate_cross_2d, collate_tokens, collate_tokens_coords
from ..model.tokenizer import UniMolTokenizer
from .masking import mask_unimol_item


class UniMolPretrainDataset(Dataset):
    """Mask a sequence of pre-featurized Uni-Mol items on the fly."""

    def __init__(
        self,
        items: Sequence[dict],
        tokenizer: Optional[UniMolTokenizer] = None,
        *,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.05,
        random_token_prob: float = 0.05,
        noise_type: str = "uniform",
        noise: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.items = [item for item in items if item]
        if not self.items:
            raise ValueError("no valid featurized items")
        self.tokenizer = tokenizer or UniMolTokenizer()
        self.mask_prob = float(mask_prob)
        self.leave_unmasked_prob = float(leave_unmasked_prob)
        self.random_token_prob = float(random_token_prob)
        self.noise_type = noise_type
        self.noise = float(noise)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return mask_unimol_item(
            self.items[index],
            self.tokenizer,
            mask_prob=self.mask_prob,
            leave_unmasked_prob=self.leave_unmasked_prob,
            random_token_prob=self.random_token_prob,
            noise_type=self.noise_type,
            noise=self.noise,
            seed=self.seed + int(index),
        )

    def collater(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        return collate_pretrain_batch(batch, self.tokenizer)


class LMDBPretrainDataset(Dataset):
    """Read featurized items from the LMDB layout produced by ``prepare-lmdb``."""

    def __init__(
        self,
        lmdb_path: str,
        tokenizer: Optional[UniMolTokenizer] = None,
        *,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.05,
        random_token_prob: float = 0.05,
        noise_type: str = "uniform",
        noise: float = 1.0,
        seed: int = 0,
    ) -> None:
        path = Path(lmdb_path)
        if not path.exists():
            raise FileNotFoundError("LMDB does not exist: %s" % path)
        self.env = lmdb.open(
            str(path),
            subdir=path.is_dir(),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        with self.env.begin() as transaction:
            self.keys = list(transaction.cursor().iternext(values=False))
        if not self.keys:
            raise ValueError("LMDB has no entries: %s" % path)
        self.tokenizer = tokenizer or UniMolTokenizer()
        self.mask_prob = float(mask_prob)
        self.leave_unmasked_prob = float(leave_unmasked_prob)
        self.random_token_prob = float(random_token_prob)
        self.noise_type = noise_type
        self.noise = float(noise)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.keys)

    def _read(self, index: int) -> dict:
        with self.env.begin() as transaction:
            raw = transaction.get(self.keys[index])
        if raw is None:
            return {}
        payload = pickle.loads(raw)
        return payload.get("item", {})

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        # ``prepare-lmdb`` skips invalid rows by default; this loop makes an
        # older LMDB with empty/error rows usable without crashing the loader.
        for offset in range(len(self.keys)):
            current = (int(index) + offset) % len(self.keys)
            item = self._read(current)
            if not item:
                continue
            return mask_unimol_item(
                item,
                self.tokenizer,
                mask_prob=self.mask_prob,
                leave_unmasked_prob=self.leave_unmasked_prob,
                random_token_prob=self.random_token_prob,
                noise_type=self.noise_type,
                noise=self.noise,
                seed=self.seed + int(index),
            )
        raise RuntimeError("LMDB contains no usable featurized items")

    def collater(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        return collate_pretrain_batch(batch, self.tokenizer)


def collate_pretrain_batch(
    batch: Sequence[Dict[str, Tensor]],
    tokenizer: Optional[UniMolTokenizer] = None,
) -> Dict[str, Tensor]:
    """Pad model inputs and pretraining targets to the same sequence length."""
    tokenizer = tokenizer or UniMolTokenizer()
    pad_idx = tokenizer.pad_token_id

    def tokens(key: str) -> Tensor:
        return collate_tokens(
            [item[key] for item in batch],
            pad_idx=pad_idx,
            left_pad=False,
            pad_to_multiple=8,
        )

    def coords(key: str) -> Tensor:
        return collate_tokens_coords(
            [item[key] for item in batch],
            pad_idx=0,
            left_pad=False,
            pad_to_multiple=8,
        )

    def pairs(key: str) -> Tensor:
        return collate_cross_2d(
            [item[key] for item in batch],
            pad_idx=0,
            left_pad=False,
            pad_to_multiple=8,
        )

    return {
        "src_tokens": tokens("src_tokens"),
        "src_coord": coords("src_coord"),
        "src_distance": pairs("src_distance"),
        "src_edge_type": pairs("src_edge_type"),
        "tokens_target": tokens("tokens_target"),
        "coord_target": coords("coord_target"),
        "distance_target": pairs("distance_target"),
    }


__all__ = [
    "LMDBPretrainDataset",
    "UniMolPretrainDataset",
    "collate_pretrain_batch",
]
