"""Write, validate and read an SMS-DataSpec container of molecule records.

Two record kinds are demonstrated: ``smiles`` (featurized on read) and
``pregenerated`` (conformer frozen at build time).  Needs the optional data
backend: ``python -m pip install -e ".[dataspec]"``.

Usage:

    python examples/container_records.py                 # writes to a temp dir
    python examples/container_records.py --output demo.msdata
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from unimol_lite.dataspec import build_record, iter_items, iter_records, validate_container, write_records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SMS-DataSpec container round-trip on molecule records")
    parser.add_argument("--output", default=None, help="container directory (default: temporary directory)")
    args = parser.parse_args(argv)

    root = Path(args.output) if args.output else Path(tempfile.mkdtemp()) / "molecules.msdata"

    records = [
        build_record("CCO", kind="smiles", name="ethanol"),
        build_record("c1ccccc1O", kind="pregenerated", name="phenol", coord_mode="auto"),
    ]
    meta = write_records(root, records, shuffle_strategy="row_group_shuffle")
    print(f"wrote {root}  counts={meta.counts.to_dict()}  fingerprint={meta.fingerprint}")

    report = validate_container(root)
    print(report.summary())
    if not report.ok:
        return 1

    for record in iter_records(root):
        print(f"record: {record['molecule_key']} kind={record['unimol_kind']} smiles={record['smiles']}")

    for record, item, status in iter_items(root):
        shapes = {key: tuple(value.shape) for key, value in item.items()} if item is not None else None
        print(f"item: {record['smiles']} status={status['status']} shapes={shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
