"""``featurize_batch`` throughput benchmark.

The built-in sample set is a **smoke set only** (no private data): it has
30+ molecules of 5-40 heavy atoms, 5 of 40-80 and one >400-atom 2D input.
For the acceptance numbers use the stratified HMDB sample instead:

    python benchmarks/bench_featurize.py \
        --csv ../data/hmdb/hmdb_compounds_20250106.csv --per-bin 25 --workers 1,4,8

``--csv`` needs a ``SMILES`` column and selects, deterministically in file order,
the first ``--per-bin`` rows per heavy-atom bin
``[5,15) [15,25) [25,40) [40,60) [60,80)``. Performance is only printed, never
asserted — CI machines are too noisy for a threshold.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # ``python benchmarks/bench_featurize.py``
    sys.path.insert(0, str(_REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from unimol_lite import featurize_batch  # noqa: E402

RDLogger.DisableLog("rdApp.*")

#: Heavy-atom bins used by the stratified ``--csv`` sampling.
BINS = ((5, 15), (15, 25), (25, 40), (40, 60), (60, 80))

#: Built-in smoke set. Only for quick local runs — **not** for acceptance.
#: 31 x 5-40 heavy atoms, 5 x 40-80 heavy atoms, 1 x >400 atoms after AddHs.
SAMPLE_SMILES = [
    # --- 5-40 heavy atoms --------------------------------------------------
    "CCCCO",
    "CC(=O)OC",
    "C1CCCCC1",
    "c1ccccc1O",
    "NCCc1ccccc1",
    "OCC(O)CO",
    "OC(=O)C(O)C(O)C(=O)O",
    "CCOC(C)=O",
    "C[C@H](N)C(=O)O",
    "OC(=O)CCCCC(=O)O",
    "c1ccc2ccccc2c1",
    "Nc1ncnc2[nH]cnc12",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "O=S(=O)(Nc1ccccn1)c1ccc(O)cc1",
    "CC(C)(C)c1ccc(cc1)C(C)(C)C",
    "C[C@H](N)Cc1ccc(O)cc1",
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "O=C(O)[C@@H](N)Cc1ccc(O)cc1",
    "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
    "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(O)(=O)OP(O)(=O)OP(O)(O)=O)[C@@H](O)[C@H]1O",
    "CC(C)C(NC(=O)C(CC1=CN=CN1)C(=O)O)C(=O)O",
    "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1",
    "O=C(Nc1ccccc1)c1ccccc1",
    "CCS(=O)(=O)N1CCN(CC1)c1ccc(Cl)cc1",
    # --- 40-80 heavy atoms -------------------------------------------------
    "[H][C@@](COC(=O)CCCC(O)\\C=C\\C=C\\C\\C=C\\CCCCCCCC)(COP(O)(O)=O)OC(=O)CCCCCCCCC\\C=C/CCCCCCCC",
    "[H][C@@](COC(=O)CCCCCCCCCCCCCCCCCCCCC)(COP(O)(=O)OCCN)OC(=O)CCC\\C=C\\C\\C=C/C[C@@H](O)\\C=C/C=C\\CCCCC",
    "[H][C@@](COC(=O)CCC(O)\\C=C/C=C\\C\\C=C/C\\C=C/C\\C=C/C\\C=C/CC)(COP(O)(=O)OCCN)OC(=O)CCCCCCC\\C=C/CCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC\\C=C/C=C/C(=O)CCCCC",
    "[H][C@](CO)(COC(=O)CCC\\C=C/C\\C=C/C\\C=C/C\\C=C/CCCCC)OC(=O)CCCCCCC\\C=C/CCCC",
    # --- >400 atoms after AddHs -> 2D path ---------------------------------
    "OCC" * 100,
]

STATUS_COLUMNS = ("ok", "degraded_2d", "invalid", "error")
STATUS_ORDER = STATUS_COLUMNS + ("timeout_error", "skipped")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", default="1,4,8", help="comma separated worker counts (default: 1,4,8)")
    parser.add_argument("--timeout", default="30.0", help="per-conformer budget in seconds (default: 30.0; 'none' disables)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N SMILES after selection")
    parser.add_argument("--csv", default=None, help="external CSV with a 'SMILES' column")
    parser.add_argument(
        "--per-bin",
        type=int,
        default=25,
        help="rows per heavy-atom bin for --csv (default: 25)",
    )
    return parser.parse_args(argv)


def _parse_timeout(text):
    text = str(text).strip().lower()
    if text in ("none", "null", "off"):
        return None
    return float(text)


def _parse_workers(text):
    workers = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise SystemExit(f"--workers must be positive integers, got {value}")
        if value not in workers:
            workers.append(value)
    if not workers:
        raise SystemExit("--workers is empty")
    return workers


def _heavy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetNumAtoms()


def _stratified_sample(pool, per_bin):
    """Deterministically take the first ``per_bin`` rows of each heavy-atom bin."""
    if per_bin <= 0:
        raise SystemExit(f"--per-bin must be positive, got {per_bin}")
    counts = {bin_: 0 for bin_ in BINS}
    picked = []
    unparsable = 0
    out_of_bins = 0
    for smiles in pool:
        size = _heavy_atoms(smiles)
        if size is None:
            unparsable += 1
            continue
        for bin_ in BINS:
            if bin_[0] <= size < bin_[1]:
                if counts[bin_] < per_bin:
                    counts[bin_] += 1
                    picked.append(smiles)
                break
        else:
            out_of_bins += 1
        if all(count >= per_bin for count in counts.values()):
            break
    return picked, counts, unparsable, out_of_bins


def _load_smiles(args):
    if args.csv:
        import pandas as pd

        frame = pd.read_csv(args.csv)
        if "SMILES" not in frame.columns:
            raise SystemExit(f"{args.csv}: missing 'SMILES' column (columns: {list(frame.columns)})")
        pool = [str(value) for value in frame["SMILES"].dropna().tolist()]
        smiles, _, unparsable, out_of_bins = _stratified_sample(pool, args.per_bin)
        source = f"{args.csv} (per_bin={args.per_bin}, out_of_bins={out_of_bins}, unparsable={unparsable})"
    else:
        smiles = list(SAMPLE_SMILES)
        source = "builtin smoke set (use --csv for acceptance)"
    if args.limit is not None:
        smiles = smiles[: args.limit]
    return smiles, source


def _bin_counts(smiles):
    counts = {bin_: 0 for bin_ in BINS}
    other = 0
    for smiles_str in smiles:
        size = _heavy_atoms(smiles_str)
        if size is None:
            other += 1
            continue
        for bin_ in BINS:
            if bin_[0] <= size < bin_[1]:
                counts[bin_] += 1
                break
        else:
            other += 1
    return counts, other


def _format_bins(smiles):
    counts, other = _bin_counts(smiles)
    parts = [f"{lo}-{hi - 1}={counts[(lo, hi)]}" for lo, hi in BINS]
    if other:
        parts.append(f"other={other}")
    return "  ".join(parts)


def _time_run(smiles, workers, timeout):
    start = time.perf_counter()
    results = featurize_batch(smiles, workers=workers, timeout=timeout)
    return results, time.perf_counter() - start


def _status_counts(results):
    counts = {}
    for _, meta in results:
        status = meta.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _print_table(rows, total, baseline_wall):
    statuses = list(STATUS_COLUMNS)
    for status in STATUS_ORDER:
        if status not in statuses and any(status in _status_counts(res) for _, _, res in rows):
            statuses.append(status)
    for _, _, res in rows:
        for status in _status_counts(res):
            if status not in statuses:
                statuses.append(status)

    header = f"{'workers':>7}  {'wall_s':>8}  {'mol/s':>8}  {'speedup':>7}"
    for status in statuses:
        header += f"  {status:>13}"
    print(header)
    print("-" * len(header))

    for workers, wall, results in rows:
        speedup = baseline_wall / wall if wall > 0 else float("inf")
        line = f"{workers:>7}  {wall:>8.3f}  {total / wall:>8.2f}  {speedup:>6.2f}x"
        counts = _status_counts(results)
        for status in statuses:
            line += f"  {counts.get(status, 0):>13}"
        print(line)


def main(argv=None):
    args = _parse_args(argv)
    smiles, source = _load_smiles(args)
    if not smiles:
        raise SystemExit("no SMILES to benchmark")
    workers = _parse_workers(args.workers)
    timeout = _parse_timeout(args.timeout)

    print(f"samples={len(smiles)}  coord_mode=auto  timeout={timeout}  source={source}")
    print(f"bins(heavy atoms): {_format_bins(smiles)}")

    baseline_results, baseline_wall = _time_run(smiles, 1, timeout)
    rows = [(1, baseline_wall, baseline_results)]
    for count in workers:
        if count == 1:
            continue
        results, wall = _time_run(smiles, count, timeout)
        rows.append((count, wall, results))

    rows.sort(key=lambda row: row[0])
    _print_table(rows, len(smiles), baseline_wall)
    print("\nspeedup is relative to the workers=1 serial baseline")
    print("Amdahl: wall >= the slowest single molecule, so speedup <= serial_total / slowest_row;")
    print("        one oversized molecule (e.g. >400-atom 2D tensorization) caps the scaling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
