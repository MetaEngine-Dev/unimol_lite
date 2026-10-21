"""PyTorch Lightning module for Uni-Mol masked-atom pretraining."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..model.config import UniMolConfig
from ..model.model import UniMolModel


class UniMolPretrainModule(LightningModule):
    """Lightning wrapper for the three Uni-Mol pretraining objectives."""

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_eps: float = 1e-6,
        warmup_ratio: float = 0.06,
        total_steps: Optional[int] = None,
        masked_token_weight: float = 1.0,
        masked_coord_weight: float = 5.0,
        masked_dist_weight: float = 10.0,
        x_norm_weight: float = 0.01,
        delta_pair_weight: float = 0.01,
        dist_mean: float = 6.312581655060595,
        dist_std: float = 3.3899264663911888,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        config_dict = dict(model_config or {})
        config_dict.update(
            {
                "mode": "train",
                "masked_token_loss": 1.0,
                "masked_coord_loss": 1.0,
                "masked_dist_loss": 1.0,
                "x_norm_loss": x_norm_weight,
                "delta_pair_repr_norm_loss": delta_pair_weight,
            }
        )
        self.config = UniMolConfig(**config_dict)
        self.model = UniMolModel(self.config)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adam_betas = tuple(adam_betas)
        self.adam_eps = float(adam_eps)
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = total_steps
        self.masked_token_weight = float(masked_token_weight)
        self.masked_coord_weight = float(masked_coord_weight)
        self.masked_dist_weight = float(masked_dist_weight)
        self.x_norm_weight = float(x_norm_weight)
        self.delta_pair_weight = float(delta_pair_weight)
        self.dist_mean = float(dist_mean)
        self.dist_std = float(dist_std)

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(
            src_tokens=batch["src_tokens"],
            src_distance=batch["src_distance"],
            src_coord=batch["src_coord"],
            src_edge_type=batch["src_edge_type"],
            encoder_masked_tokens=batch["tokens_target"].ne(self.model.padding_idx),
        )

    def _loss(self, batch: Dict[str, torch.Tensor]):
        masked = batch["tokens_target"].ne(self.model.padding_idx)
        logits, encoder_distance, encoder_coord, x_norm, delta_norm = self(batch)

        target = batch["tokens_target"][masked]
        token_loss = F.nll_loss(
            F.log_softmax(logits.float(), dim=-1),
            target,
            ignore_index=self.model.padding_idx,
            reduction="mean",
        )

        coord_target = batch["coord_target"][masked].float()
        coord_loss = F.smooth_l1_loss(
            encoder_coord[masked].float(), coord_target, beta=1.0, reduction="mean"
        )

        valid_atoms = batch["src_tokens"].ne(self.model.padding_idx)
        distance_target = batch["distance_target"][masked].float()
        distance_target = (distance_target - self.dist_mean) / self.dist_std
        masked_distance = encoder_distance[masked].float()
        masked_valid = valid_atoms[masked]
        distance_loss = F.smooth_l1_loss(
            masked_distance[masked_valid],
            distance_target[masked_valid],
            beta=1.0,
            reduction="mean",
        )

        loss = (
            self.masked_token_weight * token_loss
            + self.masked_coord_weight * coord_loss
            + self.masked_dist_weight * distance_loss
        )
        if self.x_norm_weight > 0 and x_norm is not None:
            loss = loss + self.x_norm_weight * x_norm
        if self.delta_pair_weight > 0 and delta_norm is not None:
            loss = loss + self.delta_pair_weight * delta_norm

        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            accuracy = (predicted == target).float().mean()
        return loss, {
            "loss": loss.detach(),
            "token_loss": token_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "distance_loss": distance_loss.detach(),
            "masked_accuracy": accuracy,
            "masked_tokens": masked.sum().float(),
        }

    def training_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        for key, value in metrics.items():
            self.log(
                "train_" + key,
                value,
                prog_bar=key == "loss",
                batch_size=batch["src_tokens"].size(0),
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        for key, value in metrics.items():
            self.log(
                "val_" + key,
                value,
                prog_bar=key == "loss",
                batch_size=batch["src_tokens"].size(0),
            )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        total_steps = self.total_steps or int(self.trainer.estimated_stepping_batches)
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(total_steps * self.warmup_ratio), 1)

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(
                max(total_steps - warmup_steps, 1)
            )
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def save_pretrained(self, output_dir: str) -> None:
        """Export a HuggingFace-style inference checkpoint directory."""
        self.config.mode = "infer"
        self.model.configs.mode = "infer"
        self.model.save_pretrained(output_dir)
        self.config.mode = "train"
        self.model.configs.mode = "train"
