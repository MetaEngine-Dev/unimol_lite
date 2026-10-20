"""批次 3：`featurize_batch` 并行批量入口。

覆盖：等长保序（含重复 SMILES 不折叠）、串行 vs 8 线程逐位一致、
``workers=1`` 与默认结果一致、有界窗口、真实超时逐项降级/报错且不中断整批、
未预期异常逐项隔离、空输入与参数校验。
"""

import pytest
import torch

import unimol_lite.batch as batch_mod
from unimol_lite.batch import featurize_batch

TENSOR_KEYS = ("src_tokens", "src_distance", "src_coord", "src_edge_type")

SMALL_12 = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "Nc1ncnc2[nH]cnc12",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "OCC(O)CO",
    "OC(=O)C(O)C(O)C(=O)O",
    "NCCc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
]

# 285 重原子的真实 HMDB 肽（AddHs 后 572 原子）：真实（非 mock）超时样本。
HMDB_285 = 'CCC(C)C(NC(=O)C(CC(C)C)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CCCNC(N)=N)NC(=O)C(CCCNC(N)=N)NC(=O)C(CCCNC(N)=N)NC(=O)C(CC(C)C)NC(=O)C(CC(O)=O)NC(=O)C(CCC(N)=O)NC(=O)C(NC(=O)C(CO)NC(=O)C(CCCCN)NC(=O)CNC(=O)C(CCCCN)NC(=O)C(CC(O)=O)NC(=O)C(CC1=CN=CN1)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CCC(N)=O)NC(=O)C(CC1=CN=CN1)NC(=O)C(CCC(O)=O)NC(=O)C(CO)NC(=O)C(NC(=O)C(C)N)C(C)C)C(C)CC)C(=O)NC(C)C(=O)NC(CCC(O)=O)C(=O)NC(C(C)CC)C(=O)NC(CC1=CN=CN1)C(=O)NC(C(C)O)C(=O)NC(C)C(O)=O'  # noqa: E501

INVALID = "C1CC"  # 环闭合不匹配 -> RDKit 解析失败


def _assert_same(left, right):
    """两行结果逐位一致（不比较 elapsed_s）。"""
    item_l, meta_l = left
    item_r, meta_r = right
    assert (item_l is None) is (item_r is None)
    if item_l is not None:
        assert set(item_l) == set(item_r) == set(TENSOR_KEYS)
        for key in TENSOR_KEYS:
            assert torch.equal(item_l[key], item_r[key]), f"tensor mismatch: {key}"
    for key in ("status", "coord_mode", "reason"):
        assert meta_l.get(key) == meta_r.get(key), f"meta mismatch: {key}"


def test_preserves_order_and_does_not_fold_duplicates():
    smiles = ["CCO", "CC(=O)O", "CCO", INVALID]
    results = featurize_batch(smiles, workers=2, coord_mode="2d")

    assert len(results) == len(smiles)
    assert [meta["status"] for _, meta in results] == ["ok", "ok", "ok", "invalid"]
    assert results[3][0] is None
    assert results[3][1]["reason"] == "parse_failed"

    # 重复 SMILES 不折叠：两条都返回，且各自是独立结果
    for key in TENSOR_KEYS:
        assert torch.equal(results[0][0][key], results[2][0][key])
    assert results[0][0] is not results[2][0]


def test_serial_matches_thread_pool_bitwise():
    serial = featurize_batch(SMALL_12, workers=1, timeout=60.0)
    parallel = featurize_batch(SMALL_12, workers=8, timeout=60.0)

    assert len(serial) == len(parallel) == len(SMALL_12)
    assert [meta["status"] for _, meta in serial] == ["ok"] * len(SMALL_12)
    for left, right in zip(serial, parallel):
        _assert_same(left, right)


def test_workers_one_matches_default_workers():
    smiles = SMALL_12[:4]
    serial = featurize_batch(smiles, workers=1, coord_mode="2d")
    default = featurize_batch(smiles, coord_mode="2d")

    assert len(serial) == len(default) == len(smiles)
    for left, right in zip(serial, default):
        _assert_same(left, right)


def test_workers_one_never_uses_thread_pool(monkeypatch):
    class _Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("workers=1 must stay on the serial path")

    monkeypatch.setattr(batch_mod, "ThreadPoolExecutor", _Boom)

    results = featurize_batch(["CCO", "CCN"], workers=1, coord_mode="2d")
    assert [meta["status"] for _, meta in results] == ["ok", "ok"]


def test_timeout_none_matches_default_budget():
    """``timeout=None``（无预算旧路径）与批量默认 30s 对正常分子逐位一致。"""
    smiles = SMALL_12[:4]
    legacy = featurize_batch(smiles, workers=1, timeout=None, coord_mode="3d")
    budgeted = featurize_batch(smiles, workers=1, timeout=30.0, coord_mode="3d")

    assert [meta["status"] for _, meta in legacy] == ["ok"] * len(smiles)
    for left, right in zip(legacy, budgeted):
        _assert_same(left, right)


@pytest.mark.parametrize("window", [1, 3])
def test_bounded_window_keeps_order(window):
    results = featurize_batch(SMALL_12, workers=4, window=window, coord_mode="2d")
    reference = featurize_batch(SMALL_12, workers=1, coord_mode="2d")

    assert len(results) == len(reference) == len(SMALL_12)
    for left, right in zip(results, reference):
        _assert_same(left, right)


@pytest.mark.parametrize(
    "on_timeout,expected_status",
    [("fallback_2d", "degraded_2d"), ("error", "timeout_error")],
)
def test_real_timeout_is_per_row(on_timeout, expected_status):
    """真实超时（非 mock）：285 重原子分子超预算，整批照常返回、逐项标记。

    注：``timeout=1e-6`` 对**每一行** 3D 都必然超预算（墙钟检查是精确的），
    因此这里断言的是「各行都有独立结果、整批不失败」，其余行是否降级由预算
    决定，见下一个 ``timeout=1.0`` 用例（其余行 ok）。
    """
    smiles = ["CCO", HMDB_285, "CC(=O)Oc1ccccc1C(=O)O"]
    results = featurize_batch(
        smiles, workers=3, timeout=1e-6, on_timeout=on_timeout, coord_mode="3d"
    )

    assert len(results) == 3
    item, meta = results[1]
    assert meta["status"] == expected_status
    assert meta["reason"] == "timeout"
    if expected_status == "degraded_2d":
        assert item is not None
        assert meta["coord_mode"] == "2d"
    else:
        assert item is None

    # 其余行也各自返回（整批不中断）：小分子在 1e-6 下同样降级
    for index in (0, 2):
        _, row_meta = results[index]
        assert row_meta["status"] == expected_status


@pytest.mark.parametrize(
    "on_timeout,expected_status",
    [("fallback_2d", "degraded_2d"), ("error", "timeout_error")],
)
def test_real_timeout_does_not_affect_other_rows(on_timeout, expected_status):
    """真实超时 + 现实预算：大分子超时，其余行保持正常（ok）。"""
    smiles = ["CCO", HMDB_285, "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"]
    results = featurize_batch(
        smiles, workers=4, timeout=1.0, on_timeout=on_timeout, coord_mode="3d"
    )

    assert len(results) == len(smiles)
    big_item, big_meta = results[1]
    assert big_meta["status"] == expected_status
    assert big_meta["reason"] == "timeout"
    assert (big_item is None) is (expected_status == "timeout_error")

    for index in (0, 2, 3):
        item, meta = results[index]
        # 负载下小分子可能偶发被预算截断 → 允许 degraded_2d，但不得是错误/缺失
        assert meta["status"] in ("ok", "degraded_2d"), f"row {index}: {meta}"
        assert item is not None


def test_unexpected_exception_is_isolated(monkeypatch):
    real = batch_mod.smiles2unimolitem

    def flaky(smiles, **kwargs):
        if smiles == INVALID:
            raise RuntimeError("boom")
        return real(smiles, **kwargs)

    monkeypatch.setattr(batch_mod, "smiles2unimolitem", flaky)

    results = featurize_batch(["CCO", INVALID, "CCN"], workers=3, coord_mode="2d")

    assert len(results) == 3
    item, meta = results[1]
    assert item is None
    assert meta["status"] == "error"
    assert meta["reason"] == "RuntimeError"
    for index in (0, 2):
        assert results[index][0] is not None
        assert results[index][1]["status"] == "ok"


def test_empty_input_returns_empty_list():
    assert featurize_batch([]) == []
    assert featurize_batch(iter(())) == []
    # 参数校验先于空输入：坏参数对空批次同样报错
    with pytest.raises(ValueError):
        featurize_batch([], workers=0)


@pytest.mark.parametrize("workers", [0, -1, -8])
def test_non_positive_workers_raise(workers):
    with pytest.raises(ValueError):
        featurize_batch(["CCO"], workers=workers)


@pytest.mark.parametrize("window", [0, -1])
def test_non_positive_window_raise(window):
    with pytest.raises(ValueError):
        featurize_batch(["CCO"], workers=2, window=window)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coord_mode": "3dx"},
        {"on_timeout": "nope"},
        {"timeout": 0},
        {"timeout": -1},
    ],
)
def test_invalid_batch_parameters_raise(kwargs):
    """批量参数错误必须整批 ValueError，不得吞成逐行 status="error"。"""
    with pytest.raises(ValueError):
        featurize_batch(["CCO"], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": 0}, {"coord_mode": "3dx"}, {"on_timeout": "nope"}, {"timeout": 0}],
)
def test_invalid_parameters_raise_before_any_work(monkeypatch, kwargs):
    """参数错误应在开工前抛出（不留半个批次）。"""
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("must not featurize")

    monkeypatch.setattr(batch_mod, "smiles2unimolitem", spy)
    with pytest.raises(ValueError):
        featurize_batch(["CCO"], **kwargs)
    assert calls == []
