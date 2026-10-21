# Pretraining

[中文](../zh/training.md) | **English**

This page shows how to run Uni-Mol-style masked-atom pretraining with `unimol-lite` on your own 3D conformers. The implementation follows the [official Uni-Mol pretraining code](https://github.com/deepmodeling/Uni-Mol) as a reference, but it is not the official implementation and does not distribute official pretrained weights.

## Pretraining objectives

Uni-Mol pretraining is a masked-atom modeling task. Each molecule is first converted into Uni-Mol input tensors, then masking/perturbation is applied at the atom-token, coordinate and atom-pair distance levels:

| Objective | Input | Supervision | Default weight |
| --- | --- | --- | --- |
| Masked token | Selected atom tokens become `[MASK]`, a random atom, or stay unchanged | Original atom token | `1.0` |
| Masked coordinate | Noise is added to selected atom coordinates | Original coordinates | `5.0` |
| Masked distance | Pair distances computed from noisy coordinates | Pair distances of the original coordinates | `10.0` |

The Uni-Mol `x_norm` and pair-representation norm regularizers are kept as well, both with a default weight of `0.01`. All weights are configurable; this repository does not claim that one setting is appropriate for every dataset.

## 1. Prepare data

### Input CSV

The CSV needs at least one SMILES column. The default name is `CANON_SMILES`; use `--smiles-column` for anything else:

```csv
smiles
CCO
c1ccccc1
CC(=O)OC1=CC=CC=C1C(=O)O
```

### Build the LMDB

```shell
unimol-lite prepare-lmdb \
  --input molecules.csv \
  --output molecules.lmdb \
  --smiles-column smiles \
  --nproc 8 \
  --coord-mode auto \
  --timeout 30
```

- `--coord-mode auto|3d|2d` has the same meaning as in `smiles2unimolitem`; `auto` may use 2D coordinates for large molecules.
- `--timeout` is the wall-clock budget per conformer.
- Rows that fail parsing or conformer generation are **skipped** by default. `--keep-errors` stores them with an empty `item` and an `error` field; the pretraining dataset skips empty items when reading.
- Each LMDB entry stores `{"smiles": str, "item": {...}}`; `item` follows the [featurization API](featurization.md).
- `--nproc 1` runs serially and is convenient for debugging; multi-process mode aggregates progress by row.

## 2. Configuration

Pretraining is driven by a YAML file. Command-line options override matching configuration values:

```yaml
model:
  encoder_layers: 15
  encoder_embed_dim: 512
  encoder_ffn_embed_dim: 2048
  encoder_attention_heads: 64
  max_seq_len: 512
  dropout: 0.1
  attention_dropout: 0.1
  emb_dropout: 0.1

data:
  batch_size: 16
  num_workers: 4
  mask_prob: 0.15
  leave_unmasked_prob: 0.05
  random_token_prob: 0.05
  noise_type: uniform
  noise: 1.0

optim:
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  warmup_ratio: 0.06
  masked_token_weight: 1.0
  masked_coord_weight: 5.0
  masked_dist_weight: 10.0
  x_norm_weight: 0.01
  delta_pair_weight: 0.01

trainer:
  max_epochs: 10
  max_steps: -1
  accelerator: auto
  precision: 32-true
  seed: 1
```

### Masking parameters

| Parameter | Meaning |
| --- | --- |
| `mask_prob` | Probability that an atom is selected; if none is selected, one atom is forced |
| `random_token_prob` | Probability that a selected atom becomes a random atom token |
| `leave_unmasked_prob` | Probability that a selected atom keeps its original token |
| Remaining selected atoms | Replaced with `[MASK]` |
| `noise_type` | `uniform`, `normal`, `trunc_normal` or `none` |
| `noise` | Coordinate noise scale; `uniform` samples each coordinate from `[-noise, noise]` |

> The default `0.05 + 0.05` means roughly 90% of selected atoms use `[MASK]`, not the 80/10/10 split often used by BERT. Change the configuration if you need a different ratio.

## 3. Start pretraining

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml
```

Common overrides:

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --val-lmdb validation.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml \
  --max-epochs 20 \
  --batch-size 8 \
  --lr 5e-5 \
  --device cuda \
  --precision 16-mixed \
  --num-workers 4
```

- Without `--val-lmdb`, the training LMDB is split 90/10 into train/validation. A one-entry LMDB raises an error.
- `--max-steps` is useful for smoke tests; `--resume` accepts a Lightning checkpoint.
- `--device` is passed to Lightning's `accelerator`: `cpu` for CPU, `cuda` for one GPU, other Lightning backends as supported.
- Pretraining needs the optional `train` dependencies:

```shell
pip install -e ".[train]"
```

## 4. Output

```text
runs/pretrain/
├── checkpoints/
│   ├── epoch=000-val_loss=....ckpt
│   └── last.ckpt
├── last.ckpt
└── pretrained/
    ├── config.json
    └── model.safetensors
```

`pretrained/` is a HuggingFace-style inference export: `config.json` is written with `mode` set to `infer`, and `UniMolModel.from_pretrained("runs/pretrain/pretrained")` returns `(encoder_rep, pair_rep)`.

```python
from unimol_lite.model import UniMolModel

model = UniMolModel.from_pretrained("runs/pretrain/pretrained")
model.eval()
encoder_rep, pair_rep = model(**batch)
```

## 5. Self-test

Run the bundled tiny forward/loss check without downloading data:

```shell
unimol-lite selftest
```

It featurizes `CCO`, builds a temporary 1-layer, 32-dimensional model, computes the three pretraining losses and checks that they are finite. It does not train or save weights.

## 6. Scope

- This implementation focuses on the main "masked atoms + coordinate/distance recovery" pretraining path. It does not include every augmentation, 2D-conformer combination or distributed-training setting from the official repository.
- Data quality, conformer settings and the RDKit version all affect results; regenerate the LMDB after changing data or upgrading RDKit.
- This repository is not the official Uni-Mol release and does not distribute official pretrained weights. For the official implementation and weights, see the [official repository](https://github.com/deepmodeling/Uni-Mol).
- See [References and notices](references.md) for citations and license information.
