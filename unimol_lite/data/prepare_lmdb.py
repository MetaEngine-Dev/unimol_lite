"""Build the featurized LMDB used by Uni-Mol pretraining."""

from __future__ import annotations

import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import lmdb
import pandas as pd
from tqdm import tqdm

from .prepare_unimol_item import smiles2unimolitem


def _make_item(payload: tuple[str, Optional[str], float, int]) -> dict:
    smiles, coord_mode, timeout, max_atoms = payload
    try:
        item = smiles2unimolitem(
            smiles,
            coord_mode=coord_mode or "auto",
            timeout=timeout,
            max_atoms=max_atoms,
        )
        return {"smiles": smiles, "item": item}
    except Exception as exc:  # the caller decides whether to keep or skip
        return {"smiles": smiles, "item": {}, "error": str(exc)}


def prepare_lmdb(
    input_csv: str,
    output_lmdb: str,
    *,
    smiles_column: str = "CANON_SMILES",
    nproc: int = 1,
    chunk_size: int = 1000,
    coord_mode: str = "auto",
    timeout: Optional[float] = 30.0,
    max_atoms: int = 256,
    keep_errors: bool = False,
    map_size: int = 1024**4,
    progress: bool = True,
) -> dict:
    """Featurize a CSV column and write an LMDB.

    Invalid molecules are skipped by default; set ``keep_errors=True`` to store
    them with an empty ``item`` and an ``error`` field.  Pretraining datasets
    skip empty items at read time, so an older LMDB remains usable.
    """
    source = Path(input_csv)
    if not source.exists():
        raise FileNotFoundError("input CSV does not exist: %s" % source)
    frame = pd.read_csv(source)
    if smiles_column not in frame.columns:
        raise ValueError(
            "column %r not found in %s; available: %s"
            % (smiles_column, source, ", ".join(map(str, frame.columns)))
        )
    smiles = frame[smiles_column].dropna().astype(str).drop_duplicates().tolist()
    if not smiles:
        raise ValueError("no SMILES values found in column %r" % smiles_column)

    output = Path(output_lmdb)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output), map_size=map_size, subdir=False, readonly=False)
    written = skipped = 0
    payloads = [(value, coord_mode, timeout, max_atoms) for value in smiles]
    try:
        with env.begin(write=True) as transaction:
            if nproc <= 1:
                iterator = map(_make_item, payloads)
                for item in tqdm(iterator, total=len(payloads), disable=not progress):
                    if item["item"] or keep_errors:
                        transaction.put(str(written).encode(), pickle.dumps(item))
                        written += 1
                    else:
                        skipped += 1
            else:
                with Pool(processes=nproc) as pool:
                    iterator = pool.imap(
                        _make_item, payloads, chunksize=max(1, chunk_size // nproc)
                    )
                    for item in tqdm(
                        iterator, total=len(payloads), disable=not progress
                    ):
                        if item["item"] or keep_errors:
                            transaction.put(str(written).encode(), pickle.dumps(item))
                            written += 1
                        else:
                            skipped += 1
    finally:
        env.close()
    return {
        "written": written,
        "skipped": skipped,
        "total": len(smiles),
        "path": str(output),
    }


__all__ = ["prepare_lmdb"]
