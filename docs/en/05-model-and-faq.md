# Model, migration and FAQ

[中文](../zh-CN/05-model-and-faq.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Reference

**Prerequisites:** [Quickstart](./02-quickstart.md).

## Model API

```python
# model is a UniMolModel instance
encoder_rep, pair_rep = model(**batch)
# encoder_rep: [B, L, D]    atom- and molecule-level representations
# pair_rep:    [B, L, L, D] atom-pair representations
```

- `batch` must contain `src_tokens` / `src_distance` / `src_coord` / `src_edge_type`; a single molecule without a batch dimension is promoted automatically.
- With `config.mode == "infer"` the forward pass returns `(encoder_rep, pair_rep)`; in training mode (`mode != "infer"`) it returns a tuple that also carries loss-related terms.
- A common choice is `encoder_rep[:, 0, :]` (the `[CLS]` position) as the molecule-level representation for property prediction, similarity, and other downstream tasks.

## Migrating from use_3d

`use_3d` is retained only as a compatibility parameter and emits a `DeprecationWarning` when used.

| Old | New |
| --- | --- |
| `smiles2unimolitem(smi, use_3d=False)` | `smiles2unimolitem(smi, coord_mode="auto")` |
| `smiles2unimolitem(smi, use_3d=True)` | `smiles2unimolitem(smi, coord_mode="3d")` |

If `coord_mode` is passed as well, it takes precedence and `use_3d` is ignored.

The old default `use_3d=True` used 3D for every molecule; `coord_mode="auto"` switches to 2D above 400 atoms. Pass `coord_mode="3d"` explicitly when very large molecules need 3D coordinates.

For batch processing, use `featurize_batch(smiles_list, workers=8, timeout=30.0)`. It returns results in input order with the same length as the input and reports a status per row.

## FAQ

### Cause of `degraded_2d`

When 3D embedding fails or exceeds the `timeout` budget, the default `on_timeout="fallback_2d"` degrades the coordinates to 2D. Flexible, atom-rich molecules such as long-chain lipids are more likely to trigger this. `meta["reason"]` distinguishes `timeout`, `embed_failed` and `mmff_failed`.

### Using 2D results

The model is trained on 3D conformers, so 2D is out-of-distribution input. Filter `degraded_2d` rows by `meta["status"]` or de-prioritize them downstream.

### Two runs of the same molecule differ

- Check `timeout`: with a budget, one machine or load may produce 3D and another 2D. Use `timeout=None` when strict reproducibility is required.
- Check the inputs: `coord_mode`, `rng_seed`, `max_atoms` and `coord_pool_count` must be identical.
- Check the RDKit version: coordinates and conformers are tied to it (validated with `2025.09.1`), and upgrading may change results.

### Effect of upgrading RDKit

A new RDKit version can change conformer generation and optimization, and therefore the tensors. When downstream work depends on fixed features, pin the version and regenerate existing data.

### Filtering bad rows from batch results

```python
good = [(item, meta) for item, meta in results if item is not None and meta["status"] == "ok"]
```

To keep degraded 2D rows as well, also accept `meta["status"] == "degraded_2d"`.

### Integration with an external scheduler

`smiles2unimolitem` can be placed in an external process pool, but this is usually unnecessary: `featurize_batch` already provides a thread pool, row alignment and per-row status.

### Thread count

The default is `workers=min(8, cpu)`. The benefit depends on molecule size and hardware; adding threads is not always faster.

## Development and tests

```shell
# Editable install plus dev extras
pip install -e ".[dev]"

# Full test suite
pytest tests/ -q

# Formatting and static checks
black .
isort .
flake8 .
```

**Next:** [Pretraining](./06-training.md)
