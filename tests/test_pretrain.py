from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from unimol_lite.cli import main
from unimol_lite.data import smiles2unimolitem
from unimol_lite.data.prepare_lmdb import prepare_lmdb
from unimol_lite.model import UniMolTokenizer
from unimol_lite.training.dataset import LMDBPretrainDataset, UniMolPretrainDataset
from unimol_lite.training.lightning import UniMolPretrainModule
from unimol_lite.training.masking import mask_unimol_item


def _item(smiles: str = "CCO") -> dict:
    return smiles2unimolitem(smiles, coord_mode="auto", timeout=10.0, max_atoms=32)


def test_mask_item_shapes_and_targets():
    item = _item()
    masked = mask_unimol_item(item, UniMolTokenizer(), seed=0, mask_prob=1.0, noise=0.5)
    for key in (
        "src_tokens",
        "src_coord",
        "src_distance",
        "src_edge_type",
        "tokens_target",
        "coord_target",
        "distance_target",
    ):
        assert key in masked
    assert masked["src_tokens"].shape == item["src_tokens"].shape
    assert masked["src_distance"].shape == item["src_distance"].shape
    assert masked["tokens_target"].shape == item["src_tokens"].shape
    assert masked["coord_target"].shape == item["src_coord"].shape
    assert masked["distance_target"].shape == item["src_distance"].shape
    mask = masked["tokens_target"].ne(0)
    assert mask.any()
    assert int(mask.sum()) < masked["src_tokens"].numel()


def test_pretrain_dataset_collate_and_loss():
    dataset = UniMolPretrainDataset([_item()], seed=0)
    batch = dataset.collater([dataset[0]])
    config = {
        "encoder_layers": 1,
        "encoder_embed_dim": 32,
        "encoder_ffn_embed_dim": 64,
        "encoder_attention_heads": 4,
        "max_seq_len": 32,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "emb_dropout": 0.0,
    }
    module = UniMolPretrainModule(model_config=config, total_steps=1)
    loss, metrics = module._loss(batch)
    assert torch.isfinite(loss)
    assert metrics["masked_tokens"] > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_prepare_lmdb_and_lmdb_dataset(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]}).to_csv(csv_path, index=False)
    output = tmp_path / "data.lmdb"
    result = prepare_lmdb(
        csv_path,
        output,
        smiles_column="smiles",
        nproc=1,
        timeout=10.0,
        max_atoms=32,
        progress=False,
    )
    assert result["written"] >= 1
    dataset = LMDBPretrainDataset(str(output), seed=0)
    assert len(dataset) >= 1
    batch = dataset.collater([dataset[0]])
    assert batch["src_tokens"].ndim == 2


def test_cli_selftest():
    assert main(["selftest", "--device", "cpu"]) == 0


def test_cli_featurize_prints_and_saves(tmp_path: Path, capsys):
    code = main(
        [
            "featurize",
            "--smiles",
            "CCO",
            "--timeout",
            "10",
            "--max-atoms",
            "32",
        ]
    )
    assert code == 0
    assert '"src_tokens": [5]' in capsys.readouterr().out

    output = tmp_path / "cco.npz"
    code = main(
        [
            "featurize",
            "--smiles",
            "CCO",
            "--timeout",
            "10",
            "--max-atoms",
            "32",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.is_file()
