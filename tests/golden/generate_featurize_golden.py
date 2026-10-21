"""生成 tests/golden/featurize_golden.npz（必须在改造实现之前运行）。

用法::

    python tests/golden/generate_featurize_golden.py

覆盖用例：

- cco / aspirin / hmdb_60_80 / large_2d：走 `smiles2unimolitem`，
  统一传 `use_3d=False`（等价改造后的 `coord_mode="auto"`）；
- truncate_300：直接调 `generate_unimolitem`，300 个原子 + 固定坐标，
  覆盖 >256 原子随机截断的 `src_tokens`。

npz 结构：每个 case 存 4 个张量 `<case>__<tensor>`，另存 `meta`（JSON 字符串）。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from unimol_lite.data import smiles2unimolitem  # noqa: E402
from unimol_lite.data.prepare_unimol_item import generate_unimolitem  # noqa: E402

HMDB_CSV = REPO_ROOT.parent / "data" / "hmdb" / "hmdb_compounds_20250106.csv"
OUT = Path(__file__).resolve().parent / "featurize_golden.npz"
TENSOR_KEYS = ("src_tokens", "src_distance", "src_coord", "src_edge_type")

# >256 原子随机截断路径的固定输入（确定性构造）
TRUNC_ATOMS = np.array(["C"] * 300)
TRUNC_COORDS = np.random.RandomState(0).rand(300, 3).astype(np.float64)


def pick_hmdb_smiles(min_heavy: int = 60, max_heavy: int = 80) -> str:
    """按 CSV 行序取第一个重原子数落在 [min_heavy, max_heavy] 的可解析分子（确定性）。"""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(HMDB_CSV)
    for smi in df["SMILES"].dropna():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if min_heavy <= mol.GetNumHeavyAtoms() <= max_heavy:
            return smi
    raise RuntimeError(f"no HMDB molecule with {min_heavy}-{max_heavy} heavy atoms")


def build_cases() -> list[dict]:
    hmdb_smi = pick_hmdb_smiles()
    return [
        dict(name="cco", kind="smiles", smiles="CCO"),
        dict(name="aspirin", kind="smiles", smiles="CC(=O)OC1=CC=CC=C1C(=O)O"),
        dict(name="hmdb_60_80", kind="smiles", smiles=hmdb_smi),
        dict(name="large_2d", kind="smiles", smiles="OCC" * 150),
        dict(name="truncate_300", kind="direct"),
    ]


def main() -> None:
    payload = {}
    meta = {
        "note": "generated with pre-change implementation; smiles cases use smiles2unimolitem(use_3d=False), "
        "truncate case uses generate_unimolitem with fixed inputs",
        "defaults": dict(coord_pool_count=1, random_seed=42, max_atoms=256, normalize_coord=True),
        "cases": {},
    }
    for case in build_cases():
        if case["kind"] == "smiles":
            item = smiles2unimolitem(case["smiles"], use_3d=False)
            meta["cases"][case["name"]] = dict(kind="smiles", smiles=case["smiles"], use_3d=False)
        else:
            item = generate_unimolitem(
                atoms=TRUNC_ATOMS,
                coordinates=TRUNC_COORDS,
                random_seed=42,
            )
            meta["cases"][case["name"]] = dict(
                kind="direct",
                atoms="C x 300",
                coordinates="np.random.RandomState(0).rand(300, 3)",
                random_seed=42,
            )
        for key in TENSOR_KEYS:
            payload[f"{case['name']}__{key}"] = item[key].numpy()
        meta["cases"][case["name"]]["shape"] = [list(item[k].shape) for k in TENSOR_KEYS]

    payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT}")
    for name, info in meta["cases"].items():
        print(name, info.get("smiles", ""), info["shape"])


if __name__ == "__main__":
    main()
