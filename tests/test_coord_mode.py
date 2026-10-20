"""批次 1：`coord_mode` / `use_3d` 兼容 / `return_meta` 语义。"""

import importlib

import numpy as np
import pytest

from unimol_lite.data import smiles2coords, smiles2unimolitem

s2c = importlib.import_module("unimol_lite.data.smiles2coords")


def _fail_embed(*args, **kwargs):
    raise AssertionError("EmbedMolecule must not be called")


def _large_smiles():
    return "OCC" * 150  # 450 heavy atoms -> AddHs 后远超 400


def test_force_2d_never_embeds(monkeypatch):
    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", _fail_embed)
    (atoms, coordinates), meta = smiles2coords("CCO", count=1, coord_mode="2d", return_meta=True)
    assert meta["status"] == "ok"
    assert meta["coord_mode"] == "2d"
    assert meta["reason"] is None
    assert isinstance(meta["elapsed_s"], float)
    assert len(coordinates) == 2
    assert all(np.array_equal(coordinates[0], coordinates[i]) for i in range(len(coordinates)))
    assert len(atoms) == len(coordinates[0])


def test_auto_large_molecule_uses_2d(monkeypatch):
    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", _fail_embed)
    (atoms, coordinates), meta = smiles2coords(_large_smiles(), count=1, return_meta=True)
    assert meta["coord_mode"] == "2d"
    assert len(atoms) > 400
    assert len(coordinates) == 2


def test_auto_small_molecule_uses_3d(monkeypatch):
    calls = []
    real_embed = s2c.AllChem.EmbedMolecule

    def spy(mol, **kwargs):
        calls.append(kwargs)
        return real_embed(mol, **kwargs)

    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", spy)
    (_, coordinates), meta = smiles2coords("CCO", count=1, return_meta=True)
    assert meta["coord_mode"] == "3d"
    assert len(calls) == 1
    assert len(coordinates) == 2  # 1 * 3d + 1 * 2d


def test_use_3d_true_maps_to_3d_with_warning():
    with pytest.warns(DeprecationWarning) as records:
        (_, coordinates), meta = smiles2coords("CCO", count=1, use_3d=True, return_meta=True)
    assert meta["coord_mode"] == "3d"
    assert len(coordinates) == 2
    # warning must point at the caller, not at the library internals
    assert records[0].filename == __file__


def test_positional_bool_use_3d_true(monkeypatch):
    calls = []
    real_embed = s2c.AllChem.EmbedMolecule

    def spy(mol, **kwargs):
        calls.append(kwargs)
        return real_embed(mol, **kwargs)

    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", spy)
    with pytest.warns(DeprecationWarning) as records:
        (_, coordinates), meta = smiles2coords("CCO", 1, True, return_meta=True)
    assert meta["coord_mode"] == "3d"
    assert calls
    assert len(coordinates) == 2
    assert records[0].filename == __file__


def test_positional_bool_use_3d_false(monkeypatch):
    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", _fail_embed)
    with pytest.warns(DeprecationWarning) as records:
        (_, coordinates), meta = smiles2coords(_large_smiles(), 1, False, return_meta=True)
    assert meta["coord_mode"] == "2d"
    assert len(coordinates) == 2
    assert records[0].filename == __file__


def test_smiles2unimolitem_use_3d_warning_points_at_caller():
    with pytest.warns(DeprecationWarning) as records:
        smiles2unimolitem("CCO", use_3d=False)
    assert records[0].filename == __file__


def test_use_3d_false_maps_to_auto(monkeypatch):
    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", _fail_embed)
    with pytest.warns(DeprecationWarning):
        (_, coordinates), meta = smiles2coords(_large_smiles(), count=1, use_3d=False, return_meta=True)
    assert meta["coord_mode"] == "2d"
    assert len(coordinates) == 2


def test_explicit_coord_mode_wins_over_use_3d(monkeypatch):
    monkeypatch.setattr(s2c.AllChem, "EmbedMolecule", _fail_embed)
    with pytest.warns(DeprecationWarning):
        (_, _), meta = smiles2coords("CCO", count=1, coord_mode="2d", use_3d=True, return_meta=True)
    assert meta["coord_mode"] == "2d"


def test_invalid_smiles_with_meta_does_not_raise():
    result, meta = smiles2coords("invalid_smiles", return_meta=True)
    assert result is None
    assert meta["status"] == "invalid"
    assert meta["reason"] == "parse_failed"
    assert isinstance(meta["elapsed_s"], float)


def test_invalid_smiles_without_meta_keeps_legacy_shape():
    assert smiles2coords("invalid_smiles") == [None, None]


def test_invalid_coord_mode_raises():
    with pytest.raises(ValueError):
        smiles2coords("CCO", coord_mode="2D")


def test_smiles2unimolitem_invalid_with_meta():
    item, meta = smiles2unimolitem("invalid_smiles", return_meta=True)
    assert item is None
    assert meta["status"] == "invalid"
    assert meta["reason"] == "parse_failed"
    assert "elapsed_s" in meta


def test_smiles2unimolitem_meta_ok():
    item, meta = smiles2unimolitem("CCO", coord_mode="auto", return_meta=True)
    assert meta["status"] == "ok"
    assert meta["coord_mode"] == "3d"
    assert meta["reason"] is None
    assert isinstance(meta["elapsed_s"], float)
    assert set(item) == {"src_tokens", "src_distance", "src_coord", "src_edge_type"}
