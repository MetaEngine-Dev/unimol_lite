# AGENTS.md

Instructions for AI assistants working in `unimol_lite`.

## What this repo is

UniMol-Lite is a lightweight implementation of Uni-Mol. It converts SMILES into
Uni-Mol model inputs (RDKit conformer generation + tensorization), provides a
forward-model API, masked-atom pretraining and a CLI.

It is an independent third-party implementation, **not** the official Uni-Mol
release. It does not ship official weights, training data or LMDB files.

## First, figure out what the user wants

| User asks for | What to do |
|---|---|
| "Featurize these SMILES" | Run single-molecule or batch featurization, check `status` |
| "Train/pretrain a model" | Check for SMILES/LMDB data, prepare LMDB, run pretraining |
| "Load a model" | Ask for the exported model directory; do not assume official weights |
| "Just run something" | Run `unimol-lite selftest` and label it as a smoke test |

Do not start a long pretraining run before checking the data, atom counts and
device. Do not invent outputs if no data or model is available.

## Hard rules

- Do not commit data, LMDB files, model weights or run artifacts.
- Do not fabricate data, tensors or metrics.
- Do not claim this is the official Uni-Mol implementation.
- Do not download official weights from unofficial mirrors.
- Do not replace `unimol_lite` with official Uni-Mol code.
- Keep changes small and run the checks before answering.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For pretraining:

```bash
python -m pip install -e ".[train]"
```

Verify:

```bash
python -c "import unimol_lite; print(unimol_lite.__version__)"
unimol-lite --help
unimol-lite selftest
```

There is no CI in this repo. Run the checks locally.

## Data: talk to the user first

The repo does not include data. Ask:

- Do you already have SMILES, 3D conformers, or an LMDB?
- Do you need featurization only, or masked-atom pretraining as well?
- Which atom cap and coordinate mode should be used?
- Where should the LMDB, runs and models be stored?
- Do you accept the license of any external data or weights?

Rules:

- Featurization only needs valid SMILES.
- Pretraining needs 3D conformer data or an LMDB built from SMILES.
- Do not fabricate 3D coordinates or LMDB entries.
- Do not redistribute official Uni-Mol weights.

## Featurization gate

```bash
unimol-lite featurize --smiles "CCO"
```

The output JSON reports `status`, `coord_mode` and the four tensor shapes. For
batch use, check that results are the same length as the input and inspect
`meta["status"]`:

- `ok` — normal;
- `degraded_2d` — 3D failed or timed out; out of distribution for 3D-trained models;
- `invalid` — SMILES could not be parsed;
- `timeout_error` / `skipped` — timeout handling;
- `error` — unexpected per-row exception.

## Pretraining

```bash
unimol-lite prepare-lmdb \
  --input molecules.csv \
  --output molecules.lmdb \
  --smiles-column smiles \
  --nproc 8

unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml
```

Output:

```text
runs/pretrain/pretrained/
├── config.json
└── model.safetensors
```

Load the export with:

```python
from unimol_lite.model import UniMolModel

model = UniMolModel.from_pretrained("runs/pretrain/pretrained")
model.eval()
encoder_rep, pair_rep = model(**batch)
```

## Missing dependencies

| Error | Fix |
|---|---|
| `ModuleNotFoundError: pytorch_lightning` | `python -m pip install -e ".[train]"` |
| `ModuleNotFoundError: rdkit` | Reinstall the base package; check Python 3.10/3.11 |
| CLI not found | `python -m pip install -e .` |

## Before you answer

Run what is relevant:

```bash
pytest tests/ -q
unimol-lite selftest
git status --short
```

Remove any data, LMDB, model or report artifact from the commit.

When you report results, include the command, input source, coordinate mode,
timeout, and any `degraded_2d` / `invalid` counts.

## If the user says "just give me the result"

1. Look for attached SMILES, conformer data or an LMDB. If none, ask for it.
2. If they only want to see the pipeline work, run `unimol-lite selftest` and
   label it as a demo.
3. Never make up tensors, weights or scores.

## 中文速览

先确认用户是要特征化、预训练、加载模型，还是只跑示例。本仓库不提供官方 Uni-Mol
权重或数据；预训练数据由用户提供。特征化只需要 SMILES；预训练需要 3D 构象或 LMDB。
遇到 `degraded_2d` 要说明它是分布外输入，按 `meta["status"]` 过滤。不要伪造数据、张量
或指标，不要提交数据、LMDB、模型权重或运行产物。引用与许可见 `docs/zh-CN/09-references.md`。
