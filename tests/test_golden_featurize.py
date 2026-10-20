"""批次 1：与改造前 golden 快照逐位比对。

golden 由 `tests/golden/generate_featurize_golden.py` 在改造前生成，
覆盖小/中/大分子、>400 原子 2D 路径与 >256 原子随机截断路径。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from unimol_lite.data import smiles2unimolitem
from unimol_lite.data.prepare_unimol_item import generate_unimolitem

GOLDEN = Path(__file__).resolve().parent / "golden" / "featurize_golden.npz"
TENSOR_KEYS = ("src_tokens", "src_distance", "src_coord", "src_edge_type")
CASES = ("cco", "aspirin", "hmdb_60_80", "large_2d", "truncate_300")

# 与生成脚本保持完全一致的固定输入
TRUNC_ATOMS = np.array(["C"] * 300)
TRUNC_COORDS = np.random.RandomState(0).rand(300, 3).astype(np.float64)


def _golden():
    assert GOLDEN.exists(), f"missing golden file: {GOLDEN}"
    return np.load(GOLDEN)


def _meta():
    return json.loads(str(_golden()["meta"]))


def _run_case(name):
    info = _meta()["cases"][name]
    if info["kind"] == "smiles":
        return smiles2unimolitem(info["smiles"], coord_mode="auto")
    return generate_unimolitem(atoms=TRUNC_ATOMS, coordinates=TRUNC_COORDS, random_seed=42)


@pytest.mark.parametrize("name", CASES)
def test_golden_tensors_bitwise_equal(name):
    golden = _golden()
    item = _run_case(name)
    for key in TENSOR_KEYS:
        expected = torch.from_numpy(golden[f"{name}__{key}"])
        assert torch.equal(item[key], expected), f"golden mismatch: {name}.{key}"
