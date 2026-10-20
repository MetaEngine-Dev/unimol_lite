# Model, migration and FAQ

[中文](../zh/model-and-faq.md) | **English**

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

`use_3d` is deprecated: it still works but emits a `DeprecationWarning`.

| Old | New |
| --- | --- |
| `smiles2unimolitem(smi, use_3d=False)` | `smiles2unimolitem(smi, coord_mode="auto")` |
| `smiles2unimolitem(smi, use_3d=True)` | `smiles2unimolitem(smi, coord_mode="3d")` |

If you pass `coord_mode` explicitly as well, `coord_mode` wins and `use_3d` is ignored.

The one existing downstream call site, `uni_classify/model_tools/system.py:41`, before and after:

```python
# Before
unimolitem = smiles2unimolitem(smiles, use_3d=False)

# After (same behaviour: 3D for <=400 atoms, 2D above)
unimolitem = smiles2unimolitem(smiles, coord_mode="auto")
```

> Note: the old default `use_3d=True` used 3D for every size, whereas `coord_mode="auto"` switches to 2D above 400 atoms. If you relied on 3D for very large molecules, pass `coord_mode="3d"` explicitly.

If you want to replace a dask process pool at the call site with the in-library thread pool, use `featurize_batch(smiles_list, workers=8, timeout=30.0)` (equal length, input order, per-row status).

## FAQ

### Why did my molecule become degraded_2d?

When 3D embedding fails or exceeds the `timeout` budget, `on_timeout="fallback_2d"` (the default) degrades to 2D coordinates. Flexible, atom-rich molecules such as long-chain lipids fail to embed fairly often; this is expected. Check `meta["reason"]` to tell `timeout` / `embed_failed` / `mmff_failed` apart.

### Can I treat 2D output as a normal result?

Be careful. The model was trained on 3D conformers, so 2D is out-of-distribution input. Do not silently treat `degraded_2d` as normal — filter it out or de-prioritize it by `meta["status"]`.

### Why do two runs of the same molecule differ?

- Check `timeout`: with a budget, one machine or load may produce 3D and another 2D. Use `timeout=None` when you need reproducibility.
- Check the inputs: are `coord_mode`, `rng_seed`, `max_atoms` and `coord_pool_count` identical?
- Check the RDKit version: coordinates and conformers are tied to it (validated with `2025.09.1`), and upgrading can change results.

### Does upgrading RDKit affect existing data?

Yes. A new RDKit version can change conformer generation and optimization, and therefore the tensors. If downstream work depends on fixed features, pin the version and regenerate existing data.

### How do I filter bad rows out of batch results?

```python
good = [(item, meta) for item, meta in results if item is not None and meta["status"] == "ok"]
```

To keep degraded 2D rows as well, also accept `meta["status"] == "degraded_2d"`.

### Can I wrap smiles2unimolitem in my own process pool?

Yes, but it is usually unnecessary: `featurize_batch` already runs a thread pool and handles row alignment and statuses.

### How many workers should I use?

Start with `workers=8`. The benefit depends on molecule size and hardware; more threads are not always faster.

## Development and tests

```shell
# Editable install plus dev extras
pip install -e ".[dev]"

# Full test suite
pytest tests/ -q
# 89 passed, 1 skipped

# Formatting and static checks
black .
isort .
flake8 .
```
