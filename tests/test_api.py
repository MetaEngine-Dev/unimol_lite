"""Layer 3 contract tests: single/batch agreement and the forward entry point.

The point of the layer-3 surface is that the single-molecule call, the batch call
and the model forward share one contract.  These tests pin that:
``featurize`` and ``featurize_batch`` accept the same parameters with the same
defaults and produce identical tensors, failures are reported the same way, and
``forward_features`` returns the pair of feature tensors for one item or many.
"""

from __future__ import annotations

import pytest
import torch
import unimol_lite
from unimol_lite import api
from unimol_lite.data import smiles2unimolitem

TINY_CONFIG = dict(
    encoder_layers=1,
    encoder_embed_dim=32,
    encoder_ffn_embed_dim=64,
    encoder_attention_heads=4,
    max_seq_len=32,
    dropout=0.0,
    attention_dropout=0.0,
    emb_dropout=0.0,
    activation_dropout=0.0,
)

ITEM_KEYS = ("src_tokens", "src_coord", "src_distance", "src_edge_type")


class TestFeaturizeContract:
    def test_single_and_batch_agree(self):
        item = api.featurize("CCO")
        batch_item, meta = api.featurize_batch(["CCO"], workers=1)[0]
        assert meta["status"] == "ok"
        for key in ITEM_KEYS:
            assert torch.equal(item[key], batch_item[key]), key

    def test_batch_defaults_match_single_defaults(self):
        assert api.DEFAULT_FEATURIZE_TIMEOUT == 30.0
        item, meta = api.featurize("CCO", return_meta=True)
        assert meta["status"] == "ok"
        assert meta["coord_mode"] in ("3d", "2d")
        batch_item, batch_meta = api.featurize_batch(["CCO"], workers=1)[0]
        for key in ITEM_KEYS:
            assert torch.equal(item[key], batch_item[key]), key
        assert batch_meta["coord_mode"] == meta["coord_mode"]

    def test_return_meta_shape(self):
        item, meta = api.featurize("CCO", return_meta=True)
        assert set(item) == set(ITEM_KEYS)
        assert {"status", "coord_mode", "reason", "elapsed_s"} <= set(meta)

    def test_invalid_smiles_raises_without_meta(self):
        with pytest.raises(ValueError):
            api.featurize("not-a-smiles")

    def test_invalid_smiles_is_reported_with_meta(self):
        item, meta = api.featurize("not-a-smiles", return_meta=True)
        assert item is None
        assert meta["status"] == "invalid"

    @pytest.mark.parametrize("kwargs", [{"coord_mode": "bogus"}, {"on_timeout": "bogus"}, {"timeout": -1}])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            api.featurize("CCO", **kwargs)

    def test_legacy_import_paths_still_work(self):
        from unimol_lite.batch import featurize_batch as from_batch
        from unimol_lite.data import featurize_batch as from_data

        assert from_batch is from_data
        assert callable(unimol_lite.featurize_batch)

    def test_doctest_free_low_level_contract_is_unchanged(self):
        # low-level default stays timeout=None (historical path); the standard
        # entry is the one that aligns with the batch default
        item = smiles2unimolitem("CCO")
        assert set(item) == set(ITEM_KEYS)


@pytest.fixture(scope="module")
def tiny_model():
    return api.build_model(TINY_CONFIG)


class TestForwardFeatures:
    def test_shapes_for_one_item(self, tiny_model):
        item = api.featurize("CCO")
        encoder_rep, pair_rep = api.forward_features(tiny_model, item)
        assert encoder_rep.shape == (1, item["src_tokens"].shape[0], TINY_CONFIG["encoder_embed_dim"])
        assert pair_rep.shape == (
            1,
            item["src_tokens"].shape[0],
            item["src_tokens"].shape[0],
            TINY_CONFIG["encoder_attention_heads"],
        )
        assert encoder_rep.dtype == torch.float32

    def test_batched_items_are_padded_to_a_multiple_of_eight(self, tiny_model):
        items = [api.featurize("CCO"), api.featurize("c1ccccc1")]
        encoder_rep, pair_rep = api.forward_features(tiny_model, items)
        assert encoder_rep.shape[0] == 2
        assert encoder_rep.shape[1] % 8 == 0
        assert pair_rep.shape[1] == pair_rep.shape[2] == encoder_rep.shape[1]

    def test_missing_item_key_is_refused(self, tiny_model):
        item = api.featurize("CCO")
        item.pop("src_edge_type")
        with pytest.raises(ValueError):
            api.forward_features(tiny_model, item)

    def test_training_mode_model_is_refused(self):
        training_model = api.build_model({**TINY_CONFIG, "mode": "train"})
        with pytest.raises(ValueError):
            api.forward_features(training_model, api.featurize("CCO"))

    def test_gradients_are_not_tracked(self, tiny_model):
        encoder_rep, _ = api.forward_features(tiny_model, api.featurize("CCO"))
        assert not encoder_rep.requires_grad


class TestModelEntryPoints:
    def test_build_model_is_evaluating(self):
        model = api.build_model(TINY_CONFIG)
        assert model.training is False
        assert model.configs.mode == "infer"

    def test_export_and_load_roundtrip(self, tmp_path):
        model = api.build_model(TINY_CONFIG)
        model.save_pretrained(tmp_path / "export")
        loaded = api.load_model(tmp_path / "export")
        item = api.featurize("CCO")
        first, _ = api.forward_features(model, item)
        second, _ = api.forward_features(loaded, item)
        assert torch.allclose(first, second, atol=1e-6)


class TestPackageSurface:
    def test_layer_exports_are_available(self):
        assert callable(unimol_lite.featurize)
        assert callable(unimol_lite.featurize_batch)
        assert callable(unimol_lite.build_model)
        assert callable(unimol_lite.load_model)
        assert callable(unimol_lite.forward_features)
        assert unimol_lite.UniMolModel.__name__ == "UniMolModel"
        assert unimol_lite.UniMolConfig.__name__ == "UniMolConfig"

    def test_all_is_grouped_and_importable(self):
        for name in unimol_lite.__all__:
            assert hasattr(unimol_lite, name), name
