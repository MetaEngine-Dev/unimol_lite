"""批次 2：`timeout` 预算与 `on_timeout` 三态降级。

覆盖：真实超时降级、删除 5000 次无超时重试（失败不再重试）、MMFF 纳入预算、
``error`` / ``skip`` 语义、参数校验、多构象全有或全无、``timeout=None`` 与
``timeout=60`` 逐位一致（含真实 HMDB 分子回归）。
"""

import importlib
import time

import numpy as np
import pytest
import torch

from unimol_lite.data import FeaturizationTimeout, smiles2coords, smiles2unimolitem
from unimol_lite.data.prepare_unimol_item import generate_unimolitem
from rdkit.Chem import rdDistGeom

s2c = importlib.import_module("unimol_lite.data.smiles2coords")

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
TENSOR_KEYS = ("src_tokens", "src_distance", "src_coord", "src_edge_type")

# 真实 HMDB 分子（60–140 重原子）——P1 回归：timeout=None 与 timeout=60 必须逐位一致。
# HMDB0282338 是 P1 的复现样本：maxIterations=200 时 timeout 路径 embed_failed，
# 而 timeout=None 成功（60 个样本中 2 个命中，见 04-施工日志.md）。
HMDB_60_140 = {
    "HMDB0282338": '[H][C@@](COC(=O)CC\\C=C/C\\C=C/C[C@H](O)\\C=C\\C=C/C=C/[C@H](O)C\\C=C/CC)(COP(O)(=O)OC[C@H](N)C(O)=O)OC(=O)CCCCCCCCC\\C=C/C\\C=C/CCCCC',  # 62
    "HMDB0301618": 'CCCCCC(O)CCC1C(O)CC(=O)C1CCCCCCC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(O)(=O)OP(O)(=O)OCC1OC(C(O)C1OP(O)(O)=O)N1C=NC2=C1N=CN=C2N',  # 72
    "HMDB0039476": 'CC(=O)OCC1OC(OC2C(O)COC(OC3C(O)C(C)(C)CC4C5=CCC6C7(C)CCC(OC8OC(C(O)C(O)C8OC8OC(CO)C(O)C(O)C8OC8OC(CO)C(O)C(O)C8O)C(O)=O)C(C)(CO)C7CCC6(C)C5(C)CCC34C)C2O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O',  # 100
}

# 285 重原子的真实 HMDB 肽：embed 远超 0.1 s → 真实超时（非 mock）。
HMDB_285 = 'CCC(C)C(NC(=O)C(CC(C)C)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CCCNC(N)=N)NC(=O)C(CCCNC(N)=N)NC(=O)C(CCCNC(N)=N)NC(=O)C(CC(C)C)NC(=O)C(CC(O)=O)NC(=O)C(CCC(N)=O)NC(=O)C(NC(=O)C(CO)NC(=O)C(CCCCN)NC(=O)CNC(=O)C(CCCCN)NC(=O)C(CC(O)=O)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CCC(N)=O)NC(=O)C(CC1=CN=CN1)NC(=O)C(CCC(O)=O)NC(=O)C(CO)NC(=O)C(NC(=O)C(C)N)C(C)C)C(C)CC)C(=O)NC(C)C(=O)NC(CCC(O)=O)C(=O)NC(C(C)CC)C(=O)NC(CC1=CN=CN1)C(=O)NC(C(C)O)C(=O)NC(C)C(O)=O'


def test_real_timeout_degrades_to_2d():
    """a. 真实超时：1e-6 s 预算下 embed+MMFF 必然超预算 → 2D 降级。"""
    item, meta = smiles2unimolitem(
        ASPIRIN, coord_mode="3d", timeout=1e-6, return_meta=True
    )
    assert meta["status"] == "degraded_2d"
    assert meta["reason"] == "timeout"
    assert meta["coord_mode"] == "2d"
    assert meta["elapsed_s"] < 2.0
    assert np.isfinite(item["src_coord"].numpy()).all()


def test_real_timeout_returns_2d_coordinates():
    (atoms, coordinates), meta = smiles2coords(
        ASPIRIN, count=1, coord_mode="3d", timeout=1e-6, return_meta=True
    )
    assert meta["status"] == "degraded_2d"
    assert len(coordinates) == 2  # count + 1
    assert np.all(coordinates[0][:, 2] == 0.0), "degraded output must be planar 2D"
    assert len(atoms) == len(coordinates[0])


def test_embed_failure_calls_embed_once_no_5000_retry(monkeypatch):
    """b. mock 立即返回 -1：timeout=None 只调用一次（5000 次重试已删除）。"""
    calls = []

    def fake_embed(mol, *args, **kwargs):
        calls.append((args, kwargs))
        return -1

    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", fake_embed)

    (_, coordinates), meta = smiles2coords("CCO", count=1, return_meta=True)
    assert meta["status"] == "degraded_2d"
    assert meta["reason"] == "embed_failed"
    assert len(calls) == 1
    assert not any("maxAttempts" in kwargs for _, kwargs in calls)
    assert np.all(coordinates[0][:, 2] == 0.0)


def test_embed_failure_with_timeout_also_calls_embed_once(monkeypatch):
    """b'. timeout>0 也不重试：同 params/同 seed 重试只是确定性复现同一失败。"""
    calls = []

    def fake_embed(mol, *args, **kwargs):
        calls.append((args, kwargs))
        return -1

    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", fake_embed)

    _, meta = smiles2coords("CCO", count=1, timeout=30.0, return_meta=True)
    assert meta["status"] == "degraded_2d"
    assert meta["reason"] == "embed_failed"
    assert len(calls) == 1
    assert all(isinstance(args[0], rdDistGeom.EmbedParameters) for args, _ in calls)


def test_mmff_counts_toward_budget(monkeypatch):
    """c. MMFF 慢 0.2 s + timeout=0.1 → 丢弃 3D 结果并降级。"""
    def slow_mmff(*args, **kwargs):
        time.sleep(0.2)
        return 0

    monkeypatch.setattr(s2c.AllChem, "MMFFOptimizeMolecule", slow_mmff)

    (_, coordinates), meta = smiles2coords(
        "CCO", count=1, coord_mode="3d", timeout=0.1, return_meta=True
    )
    assert meta["status"] == "degraded_2d"
    assert meta["reason"] == "timeout"
    assert meta["elapsed_s"] < 2.0
    assert np.all(coordinates[0][:, 2] == 0.0)


def test_on_timeout_error_raises_without_meta():
    """d. on_timeout="error" + return_meta=False → FeaturizationTimeout。"""
    with pytest.raises(FeaturizationTimeout):
        smiles2unimolitem(ASPIRIN, coord_mode="3d", timeout=1e-6, on_timeout="error")
    with pytest.raises(FeaturizationTimeout):
        smiles2coords("CCO", count=1, coord_mode="3d", timeout=1e-6, on_timeout="error")


def test_on_timeout_error_with_meta():
    item, meta = smiles2unimolitem(
        ASPIRIN, coord_mode="3d", timeout=1e-6, on_timeout="error", return_meta=True
    )
    assert item is None
    assert meta["status"] == "timeout_error"
    assert meta["reason"] == "timeout"
    assert isinstance(meta["elapsed_s"], float)

    result, meta = smiles2coords(
        "CCO", count=1, coord_mode="3d", timeout=1e-6, on_timeout="error", return_meta=True
    )
    assert result is None
    assert meta["status"] == "timeout_error"


def test_on_timeout_skip():
    """e. on_timeout="skip" → None / None（不带 meta 时两处统一返回 None）。"""
    assert smiles2coords("CCO", count=1, timeout=1e-6, on_timeout="skip") is None
    assert smiles2unimolitem("CCO", timeout=1e-6, on_timeout="skip") is None

    result, meta = smiles2unimolitem(
        "CCO", timeout=1e-6, on_timeout="skip", return_meta=True
    )
    assert result is None
    assert meta["status"] == "skipped"
    assert meta["reason"] == "timeout"


@pytest.mark.parametrize("value", [0, 0.0, -1, -1e-6])
def test_invalid_timeout_raises(value):
    """f. timeout 必须为 None 或 > 0。"""
    with pytest.raises(ValueError):
        smiles2coords("CCO", timeout=value)
    with pytest.raises(ValueError):
        smiles2unimolitem("CCO", timeout=value)


@pytest.mark.parametrize("value", ["1.0", True, float("nan"), float("inf")])
def test_invalid_timeout_type_raises(value):
    with pytest.raises(ValueError):
        smiles2coords("CCO", timeout=value)


@pytest.mark.parametrize("smiles", ["CCO", ASPIRIN])
def test_timeout_60_matches_none_bitwise(smiles):
    """兼容性：正常分子在 timeout=60 下不得被降级，且输出与 timeout=None 逐位一致。"""
    legacy = smiles2unimolitem(smiles, coord_mode="3d")
    budgeted = smiles2unimolitem(smiles, coord_mode="3d", timeout=60.0)
    for key in TENSOR_KEYS:
        assert torch.equal(legacy[key], budgeted[key]), f"mismatch: {smiles}.{key}"

    _, meta = smiles2unimolitem(smiles, coord_mode="3d", timeout=60.0, return_meta=True)
    assert meta["status"] == "ok"
    assert meta["reason"] is None


def test_timeout_60_generate_multi_conformer_matches_none():
    """count>1 时同样逐位一致（3D 与附加 2D 构象都不得漂移）。"""
    legacy = smiles2unimolitem("CCO", coord_pool_count=3, coord_mode="3d")
    budgeted = smiles2unimolitem("CCO", coord_pool_count=3, coord_mode="3d", timeout=60.0)
    for key in TENSOR_KEYS:
        assert torch.equal(legacy[key], budgeted[key]), f"mismatch: {key}"


def test_multi_conformer_is_all_or_nothing():
    """count>1：任一构象超时 → 丢弃全部 3D 结果（返回纯 2D，不留半套构象）。"""
    (atoms, coordinates), meta = smiles2coords(
        "CCO", count=3, coord_mode="3d", timeout=1e-6, return_meta=True
    )
    assert meta["status"] == "degraded_2d"
    assert len(coordinates) == 4  # count + 1
    assert np.all(coordinates[0][:, 2] == 0.0)
    assert all(np.array_equal(coordinates[0], c) for c in coordinates)
    # 每项都是独立副本，不共享底层 buffer
    assert len({id(c) for c in coordinates}) == len(coordinates)
    assert len(atoms) == len(coordinates[0])


def test_timeout_path_uses_embed_parameters(monkeypatch):
    """timeout>0 必须走 EmbedParameters，且参数与语义对齐旧重载。"""
    seen = []
    real_embed = s2c.AllChem.EmbedMolecule

    def spy(mol, *args, **kwargs):
        # read *before* the call: RDKit mutates maxIterations (0 -> 10 * nAtoms)
        seen.append((args, kwargs, args[0].maxIterations if args else None))
        return real_embed(mol, *args, **kwargs)

    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", spy)
    smiles2coords("CCO", count=1, coord_mode="3d", timeout=60.0)

    assert len(seen) == 1
    args, kwargs, max_iterations = seen[0]
    assert not kwargs, "timeout 路径不应使用 randomSeed= 旧重载"
    params = args[0]
    assert isinstance(params, rdDistGeom.EmbedParameters)
    # 0 == 不限次，与旧重载 maxAttempts=0 一致（P1：有限次会让难分子在 timeout 路径失败）
    assert max_iterations == 0
    assert params.randomSeed == 0
    assert params.timeout == 60
    assert params.trackFailures is True
    # 与旧重载默认一致的 ETKDG 开关（否则所有构象都会变）
    assert params.useExpTorsionAnglePrefs is True
    assert params.useBasicKnowledge is True
    assert params.useMacrocycleTorsions is True
    assert params.useMacrocycle14config is True


@pytest.mark.parametrize(
    "timeout,expected", [(0.1, 1), (1.0, 1), (1.5, 2), (30.0, 30), (1e-6, 1)]
)
def test_native_timeout_seconds(timeout, expected):
    assert s2c._native_timeout_seconds(timeout) == expected


def test_timeout_failure_count_reads_exceeded_timeout():
    """EXCEEDED_TIMEOUT 计数解析（与 RDKit 的 tuple-of-counts 布局对齐）。"""
    index = int(rdDistGeom.EmbedFailureCauses.EXCEEDED_TIMEOUT)
    with_timeout = tuple([0] * index + [2])

    class FakeParams:
        def GetFailureCounts(self):
            return with_timeout

    class EmptyParams:
        def GetFailureCounts(self):
            return ()

    assert s2c._timeout_failure_count(FakeParams()) == 2
    assert s2c._timeout_failure_count(EmptyParams()) == 0


def test_generate_unimolitem_untouched():
    """降级只发生在坐标生成层：合法 SMILES 的 item 结构不变。"""
    item = smiles2unimolitem("CCO", coord_mode="3d", timeout=60.0)
    assert set(item) == set(TENSOR_KEYS)
    assert item["src_coord"].shape == (item["src_tokens"].shape[0], 3)
    assert torch.isfinite(item["src_distance"]).all()
    # generate_unimolitem 仍是纯函数式入口
    assert callable(generate_unimolitem)


@pytest.mark.parametrize("name", sorted(HMDB_60_140))
def test_hmdb_timeout_60_matches_none_bitwise(name):
    """P1 回归：真实 HMDB 分子不得因 timeout 预算被降级或改变构象。"""
    smiles = HMDB_60_140[name]
    legacy = smiles2unimolitem(smiles, coord_mode="3d")
    budgeted, meta = smiles2unimolitem(
        smiles, coord_mode="3d", timeout=60.0, return_meta=True
    )
    assert meta["status"] == "ok"
    assert meta["reason"] is None
    for key in TENSOR_KEYS:
        assert torch.equal(legacy[key], budgeted[key]), f"mismatch: {name}.{key}"


def test_real_timeout_on_large_hmdb_molecule():
    """真实（非 mock）超时：285 重原子分子在 0.1 s 预算内超时 → 2D 降级。"""
    (atoms, coordinates), meta = smiles2coords(
        HMDB_285, count=1, coord_mode="3d", timeout=0.1, return_meta=True
    )
    assert meta["status"] == "degraded_2d"
    assert meta["reason"] == "timeout"
    assert np.all(coordinates[0][:, 2] == 0.0)
    assert len(atoms) == len(coordinates[0]) > 256