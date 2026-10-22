# Batch and runtime

[中文](../zh/batch-and-runtime.md) | **English** · [Documentation home](../../README.md#en)

## Batch semantics

`featurize_batch` is a parallel wrapper around `smiles2unimolitem` with these guarantees:

- **Equal length, input order**: one result per input, in the original order.
- **Duplicates are kept**: repeated SMILES each get their own row (no folding).
- **Per-row isolation**: an unexpected exception on one row becomes `status="error"` for that row and the batch continues.
- **Bounded submission**: a sliding window caps the number of in-flight futures instead of submitting everything at once; the input and result lists still hold `N` entries each.
- **`workers=1` is a plain serial loop**: no thread pool, no scheduling overhead.
- **Up-front validation**: invalid `coord_mode` / `on_timeout` / `timeout` / `workers` / `window` raise `ValueError` before any work is submitted (an empty input is validated too).

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `workers` | `min(8, cpu)` | thread count; `1` runs serially; `<= 0` raises `ValueError` |
| `timeout` | `30.0` | per-conformer wall-clock budget (embed + MMFF); `None` disables it |
| `on_timeout` | `"fallback_2d"` | `"fallback_2d"` / `"error"` / `"skip"` |
| `coord_mode` | `"auto"` | forwarded per row to `smiles2unimolitem` |
| `rng_seed` | `42` | seed for conformer sampling and atom truncation |
| `max_atoms` | `256` | atom cap |
| `tokenizer` | `None` | shared tokenizer (read-only); defaults to a shared instance |
| `window` | `max(workers * 4, 16)` | in-flight window size |

## Timeouts and errors

- The batch layer always calls the single-molecule API with `return_meta=True`, so a timeout under `on_timeout="error"` shows up as `status="timeout_error"` with `item=None` on that row. It **never aborts the batch**, and does not require catching `FeaturizationTimeout`.
- `on_timeout="skip"` yields `status="skipped"`; `fallback_2d` yields `status="degraded_2d"` and still returns an `item`.
- Invalid SMILES yield `status="invalid"`; unexpected exceptions yield `status="error"` with the exception type as `reason`.
- Per-row isolation means **the result list can contain rows with `item=None`**: downstream code must filter on `meta["status"]`.

## Runtime

- The default is `workers=min(8, cpu)`; the thread count depends on hardware and molecule mix;
- Large molecules may fall back to 2D when 3D embedding fails (`status="degraded_2d"`);
- Runtime depends on hardware, molecule type and thread count; run `benchmarks/bench_featurize.py` locally to measure it.

## Determinism boundaries

- Identical inputs and seeds produce identical results in serial and multi-threaded runs;
- Results depend on the RDKit version (validated with `2025.09.1`);
- With `timeout`, a slow machine may degrade a molecule to 2D; for strict reproducibility use `timeout=None`, or filter on `meta["status"]`.

## When to use the batch layer

- Large datasets with variable per-molecule cost (long-chain lipids and other hard-to-embed molecules) benefit from `featurize_batch(..., timeout=30.0)`, which keeps a bad molecule from stalling the whole run.
- For per-row auditing (which molecules degraded to 2D, which failed to parse), `meta` provides the status without additional exception handling.
- For a small number of molecules, or when the caller already has a scheduler, call `smiles2unimolitem` directly.
