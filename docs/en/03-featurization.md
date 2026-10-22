# Featurization API

[中文](../zh-CN/03-featurization.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 2 — Use

**Prerequisites:** [Installation](./01-installation.md).

## Overview

The featurization entry points live in `unimol_lite.data` (`featurize_batch` is also exported from the top-level package):

| Function | Purpose |
| --- | --- |
| `smiles2coords` | SMILES → atom symbols + coordinate list |
| `smiles2unimolitem` | SMILES → model input dict (plus a status `meta`) |
| `generate_unimolitem` | atoms + coordinates → model input dict |
| `featurize_batch` | parallel batch featurization with per-row status |
| `FeaturizationTimeout` | timeout exception (raised only for specific parameter combinations) |

## smiles2coords

```python
smiles2coords(
    smiles,
    count=10,
    coord_mode="auto",
    return_meta=False,
    use_3d=_UNSET,
    *,
    timeout=None,
    on_timeout="fallback_2d",
)
```

- Returns `(atoms, coordinates_list)`. `count` is the number of 3D conformers to generate (default `10`), so the list holds `count + 1` arrays (10 × 3D + 1 × 2D).
- With `return_meta=True` it returns `(result, meta)`.
- Invalid SMILES: `return_meta=True` gives `(None, meta)` with `status="invalid"`; `return_meta=False` gives `[None, None]`.
- Failures and timeouts are handled according to `on_timeout` (see below).

## smiles2unimolitem

```python
smiles2unimolitem(
    smiles,
    tokenizer=UniMolTokenizer(),
    coord_pool_count=1,
    random_seed=42,
    max_atoms=256,
    normalize_coord=True,
    use_3d=_UNSET,
    *addl_seeds,
    coord_mode=_UNSET,
    rng_seed=None,
    return_meta=False,
    timeout=None,
    on_timeout="fallback_2d",
)
```

Returns the model input dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `src_tokens` | `[L]` | atom tokens with `[CLS]` / `[SEP]`; hydrogens removed |
| `src_coord` | `[L, 3]` | coordinates; first and last rows are zero-padded `[CLS]` / `[SEP]` |
| `src_distance` | `[L, L]` | pairwise distance matrix |
| `src_edge_type` | `[L, L]` | pairwise edge types |

Notable parameters:

- `coord_pool_count` generates several conformers and samples one with `random_seed`; `rng_seed` sets the seed for all random draws independently (defaults to `random_seed`).
- `max_atoms` caps the atom count; excess atoms are subsampled with a local RNG.
- `normalize_coord=True` centers the coordinates on their mean.
- `use_3d` is deprecated; see the [migration notes](05-model-and-faq.md).
- `return_meta=True` returns `(item | None, meta)`; the default `False` returns just `item`.

Return conventions with `return_meta=False`:

- Success: return `item`.
- Invalid SMILES: raise `TypeError`.
- Timeout with `on_timeout="error"`: raise `FeaturizationTimeout`.
- Timeout with `on_timeout="skip"`: return `None`.
- `on_timeout="fallback_2d"`: return the degraded 2D `item` (no exception when meta is not requested).

## generate_unimolitem

```python
generate_unimolitem(
    atoms,
    coordinates,
    tokenizer=UniMolTokenizer(),
    random_seed=42,
    max_atoms=256,
    normalize_coord=True,
    *addl_seeds,
)
```

Use this when atoms and coordinates are already available: it strips hydrogens, truncates to `max_atoms`, optionally centers the coordinates and builds the same four tensors as `smiles2unimolitem`. It performs no conformer generation, so it needs no RDKit embedding and has no `timeout`.

## featurize_batch

```python
featurize_batch(
    smiles,
    *,
    workers=None,
    timeout=30.0,
    on_timeout="fallback_2d",
    coord_mode="auto",
    rng_seed=42,
    max_atoms=256,
    tokenizer=None,
    window=None,
)
```

Returns `list[(item | None, meta)]`, the same length as the input and in input order. Semantics, parameter validation and parallelism are covered in [batch and runtime](04-batch-and-runtime.md).

## coord_mode

| Value | Behaviour |
| --- | --- |
| `auto` (default) | 3D for ≤400 atoms, 2D for >400 atoms |
| `3d` | force 3D for every size; failures/timeouts follow `on_timeout` |
| `2d` | force 2D; never calls `EmbedMolecule`, no conformer generation cost |

`auto` counts atoms after adding hydrogens and matches the old `use_3d=False` inference path.

## timeout and on_timeout

`timeout` is a **per-conformer** wall-clock budget covering RDKit embedding and MMFF optimization:

- `None` disables the budget and applies no timeout; the single-molecule APIs default to `None`, the batch layer defaults to `30.0` seconds.
- RDKit's native timeout only accepts whole seconds (sub-second values round up to 1 s); the exact budget is enforced by a wall-clock check after embed/MMFF.
- With `count > 1` (`smiles2coords`) the total budget is `count * timeout`, and any single failing conformer discards all 3D results.
- Failures are not retried: the same parameters plus the same seed are deterministic, so a retry only doubles the cost.

`on_timeout` decides what happens after a timeout (or a failed 3D embedding):

| Value | Behaviour | `status` | `item` |
| --- | --- | --- | --- |
| `fallback_2d` (default) | degrade to 2D | `degraded_2d` | dict |
| `error` | report an error (`return_meta=True` records the status) | `timeout_error` | `None` |
| `skip` | skip the molecule | `skipped` | `None` |

## The meta status model

`meta` is a dict with `status` / `coord_mode` / `reason` / `elapsed_s`:

| `status` | Meaning | `item` |
| --- | --- | --- |
| `ok` | normal (3D or 2D mode) | dict |
| `degraded_2d` | 3D failed or timed out; degraded to 2D | dict |
| `timeout_error` | timeout with `on_timeout="error"` | `None` |
| `invalid` | SMILES could not be parsed or processed | `None` |
| `skipped` | timeout with `on_timeout="skip"` | `None` |
| `error` | unexpected exception caught by the batch layer | `None` |

- `coord_mode`: the mode actually used (`"2d"` after degradation); `None` when the SMILES could not be parsed (unless `"3d"` / `"2d"` was passed explicitly, in which case that value is reported). This is the **actual** mode, not the requested one.
- `reason`: `timeout` / `embed_failed` / `mmff_failed` / `parse_failed` / an exception type name.
- `elapsed_s`: wall-clock time for this call, in seconds.

## FeaturizationTimeout

`FeaturizationTimeout` derives from `RuntimeError` and is raised only when `on_timeout="error"` and `return_meta=False`. With `return_meta=True` the same condition returns `(None, meta)` with `status="timeout_error"`. The batch layer always records per-row status and never raises for a single-row timeout.

## Determinism

All randomness comes from a local `np.random.RandomState` created per call; the global `np.random` state is never touched, so identical inputs and seeds are identical in serial and multi-threaded runs. Results are tied to the RDKit version (validated with `2025.09.1`). With `timeout` enabled, the same molecule may embed in 3D on one machine (or under one load) and degrade to 2D on another. For reproducible output use `timeout=None`, or filter degraded rows by `meta["status"]`.

**Next:** [Batch and runtime](./04-batch-and-runtime.md)
