"""Command line interface for unimol_lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import __version__


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unimol-lite",
        description="Uni-Mol featurization and masked-atom pretraining tools.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-lmdb", help="featurize a CSV and write an LMDB")
    prepare.add_argument("--input", "-i", required=True, help="input CSV")
    prepare.add_argument("--output", "-o", required=True, help="output LMDB file")
    prepare.add_argument("--smiles-column", default="CANON_SMILES")
    prepare.add_argument("--nproc", type=_positive_int, default=1)
    prepare.add_argument("--chunk-size", type=_positive_int, default=1000)
    prepare.add_argument("--coord-mode", choices=["auto", "3d", "2d"], default="auto")
    prepare.add_argument("--timeout", type=float, default=30.0)
    prepare.add_argument("--max-atoms", type=_positive_int, default=256)
    prepare.add_argument("--keep-errors", action="store_true")
    prepare.add_argument("--quiet", action="store_true")

    pretrain = sub.add_parser(
        "pretrain", help="run masked-atom pretraining with Lightning"
    )
    pretrain.add_argument("--lmdb", required=True, help="training LMDB")
    pretrain.add_argument("--val-lmdb", default=None, help="optional validation LMDB")
    pretrain.add_argument("--output-dir", required=True)
    pretrain.add_argument("--config", default=None, help="YAML configuration file")
    pretrain.add_argument("--max-epochs", type=_positive_int, default=None)
    pretrain.add_argument("--max-steps", type=_positive_int, default=None)
    pretrain.add_argument("--batch-size", type=_positive_int, default=None)
    pretrain.add_argument("--lr", type=float, default=None)
    pretrain.add_argument("--weight-decay", type=float, default=None)
    pretrain.add_argument("--warmup-ratio", type=float, default=None)
    pretrain.add_argument("--num-workers", type=int, default=None)
    pretrain.add_argument(
        "--device", default=None, help="cpu, cuda, or an accelerator name"
    )
    pretrain.add_argument("--precision", default=None)
    pretrain.add_argument("--seed", type=int, default=None)
    pretrain.add_argument("--resume", default=None, help="checkpoint path")
    pretrain.add_argument("--quiet", action="store_true")

    selftest = sub.add_parser("selftest", help="run a small local smoke test")
    selftest.add_argument("--device", default="cpu")
    return parser


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pretraining configuration requires pyyaml; install with '.[train]'"
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config file must contain a YAML mapping")
    return data


def _run_prepare(args) -> int:
    from .data.prepare_lmdb import prepare_lmdb

    result = prepare_lmdb(
        args.input,
        args.output,
        smiles_column=args.smiles_column,
        nproc=args.nproc,
        chunk_size=args.chunk_size,
        coord_mode=args.coord_mode,
        timeout=args.timeout,
        max_atoms=args.max_atoms,
        keep_errors=args.keep_errors,
        progress=not args.quiet,
    )
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False))
    return 0


def _run_pretrain(args) -> int:
    try:
        import pytorch_lightning as pl
        import torch
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise ImportError(
            "pretraining requires pytorch-lightning; install with '.[train]'"
        ) from exc

    from .training.dataset import LMDBPretrainDataset
    from .training.lightning import UniMolPretrainModule

    config = _load_yaml(args.config)
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    optim_cfg = config.get("optim", {}) or {}
    trainer_cfg = config.get("trainer", {}) or {}

    def pick(cli_value, section, key, default):
        if cli_value is not None:
            return cli_value
        return section.get(key, default)

    batch_size = int(pick(args.batch_size, data_cfg, "batch_size", 16))
    num_workers = int(pick(args.num_workers, data_cfg, "num_workers", 0))
    seed = int(pick(args.seed, trainer_cfg, "seed", 1))
    learning_rate = float(pick(args.lr, optim_cfg, "learning_rate", 1e-4))
    weight_decay = float(pick(args.weight_decay, optim_cfg, "weight_decay", 1e-4))
    warmup_ratio = float(pick(args.warmup_ratio, optim_cfg, "warmup_ratio", 0.06))
    max_epochs = int(pick(args.max_epochs, trainer_cfg, "max_epochs", 10))
    max_steps = int(pick(args.max_steps, trainer_cfg, "max_steps", -1))
    precision = pick(args.precision, trainer_cfg, "precision", "32-true")
    accelerator = pick(args.device, trainer_cfg, "accelerator", "auto")

    pl.seed_everything(seed, workers=True)
    dataset = LMDBPretrainDataset(
        args.lmdb,
        mask_prob=float(data_cfg.get("mask_prob", 0.15)),
        leave_unmasked_prob=float(data_cfg.get("leave_unmasked_prob", 0.05)),
        random_token_prob=float(data_cfg.get("random_token_prob", 0.05)),
        noise_type=str(data_cfg.get("noise_type", "uniform")),
        noise=float(data_cfg.get("noise", 1.0)),
        seed=seed,
    )
    if args.val_lmdb:
        val_dataset = LMDBPretrainDataset(
            args.val_lmdb,
            mask_prob=dataset.mask_prob,
            leave_unmasked_prob=dataset.leave_unmasked_prob,
            random_token_prob=dataset.random_token_prob,
            noise_type=dataset.noise_type,
            noise=dataset.noise,
            seed=seed + 1,
        )
        train_dataset = dataset
    else:
        if len(dataset) < 2:
            raise ValueError(
                "pretraining needs at least two LMDB entries or --val-lmdb"
            )
        val_size = max(1, int(round(len(dataset) * 0.1)))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collater,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collater,
    )

    module = UniMolPretrainModule(
        model_config=model_cfg,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        total_steps=max_steps if max_steps > 0 else None,
        masked_token_weight=float(optim_cfg.get("masked_token_weight", 1.0)),
        masked_coord_weight=float(optim_cfg.get("masked_coord_weight", 5.0)),
        masked_dist_weight=float(optim_cfg.get("masked_dist_weight", 10.0)),
        x_norm_weight=float(optim_cfg.get("x_norm_weight", 0.01)),
        delta_pair_weight=float(optim_cfg.get("delta_pair_weight", 0.01)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        enable_checkpointing=True,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    module.save_pretrained(str(output_dir / "pretrained"))
    trainer.save_checkpoint(str(output_dir / "last.ckpt"))
    if not args.quiet:
        print("pretraining finished; exported %s" % (output_dir / "pretrained"))
    return 0


def _run_selftest(args) -> int:
    import torch

    from .data import smiles2unimolitem
    from .training.dataset import UniMolPretrainDataset
    from .training.lightning import UniMolPretrainModule

    item = smiles2unimolitem("CCO", coord_mode="auto", timeout=10.0)
    dataset = UniMolPretrainDataset([item], seed=0)
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
    module.to(args.device)
    batch = {key: value.to(args.device) for key, value in batch.items()}
    loss, metrics = module._loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("selftest loss is not finite")
    print(
        json.dumps(
            {key: float(value) for key, value in metrics.items()}, ensure_ascii=False
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-lmdb":
            return _run_prepare(args)
        if args.command == "pretrain":
            return _run_pretrain(args)
        if args.command == "selftest":
            return _run_selftest(args)
        parser.error("unknown command")
    except (FileNotFoundError, ValueError, ImportError, RuntimeError, OSError) as exc:
        print("unimol-lite: error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
