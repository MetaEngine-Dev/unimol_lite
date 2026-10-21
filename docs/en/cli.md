# Command-line interface

[中文](../zh/cli.md) | **English**

After installation, use the `unimol-lite` command. Without installation you can also run `python -m unimol_lite.cli`.

```shell
unimol-lite --help
unimol-lite prepare-lmdb --help
unimol-lite pretrain --help
unimol-lite selftest --help
```

## Commands

| Command | Purpose | Needs `[train]` extras |
| --- | --- | --- |
| `prepare-lmdb` | Featurize SMILES from a CSV into an LMDB | No |
| `pretrain` | Masked-atom pretraining with PyTorch Lightning | Yes |
| `selftest` | Featurize + tiny forward/loss check | No |

## prepare-lmdb

```shell
unimol-lite prepare-lmdb \
  --input molecules.csv \
  --output molecules.lmdb \
  --smiles-column smiles \
  --nproc 8 \
  --coord-mode auto \
  --timeout 30 \
  --max-atoms 256
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input`, `-i` | required | Input CSV |
| `--output`, `-o` | required | Output LMDB file |
| `--smiles-column` | `CANON_SMILES` | SMILES column name |
| `--nproc` | `1` | Featurization processes; `1` means serial |
| `--chunk-size` | `1000` | Chunk-size hint for multiprocessing |
| `--coord-mode` | `auto` | `auto` / `3d` / `2d` |
| `--timeout` | `30.0` | Per-conformer wall-clock budget in seconds |
| `--max-atoms` | `256` | Maximum atoms per molecule |
| `--keep-errors` | off | Keep failed rows (empty `item`, with `error`) |
| `--quiet` | off | Do not print the summary JSON |

On success it exits with code `0` and prints:

```json
{"written": 1234, "skipped": 5, "total": 1239, "path": "molecules.lmdb"}
```

## pretrain

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml
```

| Option | Meaning |
| --- | --- |
| `--lmdb` | Training LMDB (required) |
| `--val-lmdb` | Validation LMDB; omitted means a 90/10 split of the training set |
| `--output-dir` | Output directory (required) |
| `--config` | YAML configuration |
| `--max-epochs` | Maximum epochs |
| `--max-steps` | Maximum optimizer steps; `-1` means epoch-based |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--weight-decay` | AdamW weight decay |
| `--warmup-ratio` | Linear warmup fraction of total steps |
| `--num-workers` | DataLoader workers |
| `--device` | `cpu`, `cuda`, or a Lightning accelerator |
| `--precision` | For example `32-true`, `16-mixed` |
| `--seed` | Random seed |
| `--resume` | Resume from a checkpoint |
| `--quiet` | Suppress the completion message |

Command-line values override matching YAML values. See [Pretraining](training.md) for configuration fields and output layout.

## selftest

```shell
unimol-lite selftest
unimol-lite selftest --device cuda
```

The self-test:

1. featurizes `CCO` with RDKit;
2. builds a temporary 1-layer, 32-dimensional `UniMolModel`;
3. computes masked token / coordinate / distance losses;
4. prints each loss and checks that all are finite.

It does not train or write files. It is a quick installation and forward-pass check.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | Bad arguments, missing file, missing dependency, or runtime error |

Errors go to stderr, for example:

```text
unimol-lite: error: pretraining requires pytorch-lightning; install with '.[train]'
```
