"""Shortest end-to-end path: model -> featurize -> forward -> molecule features.

Without ``--model-dir`` a randomly initialised model is built: the pipeline runs
end to end, but the numbers are meaningless because this repository ships no
weights.  With ``--model-dir`` (a directory exported by ``unimol-lite pretrain``)
the features come from a trained model.

The features printed here are *pre-alignment* molecule features.  Turning them
into vectors that can be searched against spectra (the LayerNorm step) is the job
of the solution package that consumes this library, not of this script.

Usage:

    python examples/minimal_features.py
    python examples/minimal_features.py --model-dir runs/pretrain/pretrained --smiles CCO c1ccccc1O
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from unimol_lite import build_model, featurize, forward_features, load_model

#: Small architecture, only used for the random-weight structure demo.
DEMO_CONFIG = {
    "encoder_layers": 1,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 4,
    "max_seq_len": 32,
}

DEFAULT_SMILES = ["CCO", "c1ccccc1O"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="shortest unimol_lite featurization + forward path")
    parser.add_argument("--model-dir", default=None, help="exported model directory (omit for random weights)")
    parser.add_argument("--smiles", nargs="+", default=DEFAULT_SMILES, help="molecules to featurize")
    parser.add_argument("--coord-mode", default="auto", choices=["auto", "3d", "2d"])
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu / cuda")
    args = parser.parse_args(argv)

    if args.model_dir:
        model = load_model(args.model_dir, device=args.device)
        weights = f"weights: {args.model_dir}"
    else:
        model = build_model(DEMO_CONFIG, device=args.device)
        weights = "weights: random initialisation (structure demo only)"

    items: List[dict] = []
    kept: List[str] = []
    for smiles in args.smiles:
        item, meta = featurize(smiles, coord_mode=args.coord_mode, return_meta=True)
        if item is None:
            print(f"{smiles}: status={meta['status']} reason={meta.get('reason')}")
            continue
        atoms = int(item["src_tokens"].shape[0]) - 2  # minus CLS and SEP
        print(f"{smiles}: status={meta['status']} coord_mode={meta['coord_mode']} atoms={atoms}")
        kept.append(smiles)
        items.append(item)

    if not items:
        print("no featurizable molecules")
        return 1

    encoder_rep, pair_rep = forward_features(model, items, device=args.device)
    print(f"\n{weights}")
    print(f"encoder_rep: {tuple(encoder_rep.shape)}   pair_rep: {tuple(pair_rep.shape)}")
    print("CLS row per molecule (molecule-level candidate, pre-alignment):")
    for smiles, vector in zip(kept, encoder_rep[:, 0, :]):
        print(f"  {smiles}: dim={int(vector.shape[0])}")
    print("\nNext step for searchable vectors: the consuming solution package applies its")
    print("alignment LayerNorm (and the metric contract) to the CLS row.")
    return 0 if len(kept) == len(list(args.smiles)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
