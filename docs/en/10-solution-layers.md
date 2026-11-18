# Solution layers (L1–L4)

[中文](../zh-CN/10-solution-layers.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Concepts

**Prerequisites:** [Installation](./01-installation.md).

**Next:** [Featurization API](./03-featurization.md) · [Data containers](./11-dataspec.md)

## Why layers

UniMol-Lite is a *solution*, not just a model: it ships the network, the
pretraining recipe, the standard call surface and a shortest working path. Keeping
those four apart is what makes the standard call surface meaningful — a consumer
that reaches into an inner encoder instead of the standard call silently inherits
a different semantics (different defaults, different coordinate mode, different
post-processing) without anything failing. The layer rules below make that kind of
bypass visible and, in review, refusable.

## The four layers

| Layer | What it defines | Entry points | Boundary |
| --- | --- | --- | --- |
| **L1 Architecture** | The network and its configuration | `unimol_lite.model`: `UniMolModel`, `UniMolConfig`, `UniMolTokenizer`; encoder parts under `unimol_lite.model.layers` (internal) | Defines the forward pass only — no data preparation, no training loop, no I/O |
| **L2 Training** | Masked-atom pretraining | `unimol_lite.training`: `UniMolPretrainModule`, `LMDBPretrainDataset` / `UniMolPretrainDataset`, `mask_unimol_item`, `collate_pretrain_batch`; CLI `unimol-lite prepare-lmdb` / `pretrain` / `selftest` | Produces a model export; never produces search vectors. Requires the `train` extra |
| **L3 Standard calls** | The contract applications use | `unimol_lite.api`: `featurize`, `featurize_batch`, `build_model`, `load_model`, `forward_features`; data I/O in `unimol_lite.dataspec` | **The only layer an application should call.** Output is molecule features, pre-alignment |
| **L4 Minimal use** | The shortest end-to-end path | [`examples/minimal_features.py`](../../examples/minimal_features.py) and the snippet below | Uses L3 only; adds no logic of its own |

### Dependency rule (one direction)

```text
L4  minimal use ─┐
                 ├─> L3 standard calls ─> L1 architecture
L2  training  ───┘                         (L2 also only touches L1)
applications ─────> L3 only
```

- L3 does not reach into `unimol_lite.training`; L4 does not reach into L1 or L2.
- An application (an alignment/search package, a service, a GUI) calls L3 and
  nothing below it. If a use case seems to need an inner encoder, the fix belongs
  in L3 — that is exactly what `forward_features` is for.

## Responsibility boundary: molecule features, pre-alignment

`unimol_lite` produces **molecule features**: atom-level encoder states
`(batch, atoms + 2, dim)` plus pair states `(batch, L, L, heads)`.

It deliberately does **not**:

- apply a LayerNorm to turn a molecule vector into a spectrum-comparable one —
  that is the cross-modal alignment step of the consuming solution package (for a
  CLIP-style model: the `mol_ln` on the CLS row);
- embed, hashing-normalise or otherwise process spectra;
- own a vector store, a metric choice, or a search entry point.

Rationale: a searchable vector must be produced by exactly one implementation,
and that implementation must hold both modalities. The alignment layer owns the
LN and the metric contract; this library owns featurization and the encoder.

## L3 contracts

### Featurization

| Call | Input | Output | Notes |
| --- | --- | --- | --- |
| `featurize(smiles, ...)` | one SMILES string | item dict, or `(item \| None, meta)` with `return_meta=True` | Raises `ValueError` for unfeaturizable input when `return_meta=False` |
| `featurize_batch(smiles, ...)` | iterable of SMILES | `list[(item \| None, meta)]` — same length and order as the input | Per-row failures never abort the batch |

Item tensors (per molecule, `L = atoms + 2` for the CLS/SEP tokens):

| Key | Shape | dtype | Unit |
| --- | --- | --- | --- |
| `src_tokens` | `(L,)` | int64 | token ids; hydrogens are dropped before tokenization |
| `src_coord` | `(L, 3)` | float32 | Ångström, centred on the coordinate mean; CLS/SEP rows are zero |
| `src_distance` | `(L, L)` | float32 | Ångström, pairwise Euclidean |
| `src_edge_type` | `(L, L)` | int64 | `token_i * vocab + token_j` |

Shared parameters and defaults (identical for both calls):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `coord_mode` | `"auto"` | `"auto"` → 3D up to 400 atoms, else 2D; `"3d"`; `"2d"` |
| `timeout` | `30.0` s | per-conformer wall clock (embed + force field); `None` disables |
| `on_timeout` | `"fallback_2d"` | `"fallback_2d"` / `"error"` / `"skip"` |
| `rng_seed` | `42` | conformer choice and atom truncation |
| `max_atoms` | `256` | atom cap; larger molecules are randomly truncated |
| `workers` | `min(8, cpus)` | batch only; `1` = serial |

`meta` reports `status` (`ok` / `degraded_2d` / `invalid` / `timeout_error` /
`skipped` / `error`), `coord_mode`, `reason` and `elapsed_s`. Treat
`degraded_2d` as out-of-distribution for a 3D-trained model.

The low-level `unimol_lite.data.smiles2unimolitem` keeps its historical defaults
(`timeout=None`) for compatibility; the standard call is `unimol_lite.api.featurize`,
whose budget matches the batch default.

### Model

| Call | Input | Output |
| --- | --- | --- |
| `build_model(config=None, device=None)` | config dict or `UniMolConfig` | model with **random** weights, inference mode (structure/demo use) |
| `load_model(path, device=None)` | directory exported by pretraining (`config.json` + weights) | model in inference mode |
| `forward_features(model, items, device=None)` | one item or a list of items | `(encoder_rep, pair_rep)`: `(B, L, D)` and `(B, L, L, heads)`, float32, no gradients |

`forward_features` pads a list of items to a multiple of 8 (`L` is the padded
length) and refuses a model whose config is in training mode, because such a
model returns loss heads instead of the pair representation.

### Data containers

Reading and writing `SMS-DataSpec` parquet-block containers is part of L3:
see [Data containers](./11-dataspec.md).

## L2 note: pretraining data stays upstream-compatible

The pretraining LMDB layout follows the upstream Uni-Mol project and is
deliberately **exempt** from the container protocol: it is an internal training
artifact, not an exchange format, and changing it would break compatibility with
upstream tooling. The container protocol covers molecule/spectrum datasets (see
[Data containers](./11-dataspec.md)).

## L4: the shortest working path

```python
from unimol_lite import build_model, featurize, forward_features

# no weights shipped: this is a randomly initialised model, i.e. a structure demo
model = build_model({
    "encoder_layers": 1, "encoder_embed_dim": 32, "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 4, "max_seq_len": 32,
})
item = featurize("CCO")                      # model input tensors
encoder_rep, pair_rep = forward_features(model, item)
print(encoder_rep.shape, pair_rep.shape)     # (1, L, 32) (1, L, L, 4)
```

With a trained/exported model, replace `build_model(...)` with
`load_model("runs/pretrain/pretrained")`. The CLS row `encoder_rep[:, 0, :]` is
the conventional molecule-level vector; turning it into a searchable vector is the
alignment layer's job, not this library's.

The same path is runnable as a script:

```bash
python examples/minimal_features.py                                  # random weights
python examples/minimal_features.py --model-dir runs/pretrain/pretrained --smiles CCO c1ccccc1O
```

## Compatibility

Existing import paths keep working: `unimol_lite.data.smiles2unimolitem`,
`unimol_lite.data.featurize_batch`, `unimol_lite.batch.featurize_batch` and
`unimol_lite.model.UniMolModel` are unchanged. `unimol_lite.__all__` is grouped by
layer; `unimol_lite.dataspec` stays a submodule because it needs the optional
`dataspec` extra.
