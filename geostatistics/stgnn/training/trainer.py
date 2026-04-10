"""
InductiveTrainer: training loop for the run-based STGNN.

Logs per epoch: loss, RMSE, R² — for both train and val.
NaN values in ground truth (missing measurements) are masked out of all metrics.
"""
from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..config import ModelConfig
from ..model import STGNN
from .losses import build_loss
from .sampler import SampleBatch, TrainingSampler

logger = logging.getLogger(__name__)


def _metrics(preds: Tensor, targets: Tensor) -> tuple[float, float, float]:
    """Compute MSE, RMSE and R² over all predictions."""
    mse    = ((preds - targets) ** 2).mean().item()
    rmse   = math.sqrt(mse)
    ss_res = ((preds - targets) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, rmse, r2


class InductiveTrainer:
    def __init__(
        self,
        model: STGNN,
        sampler: TrainingSampler,
        config: ModelConfig,
        device: torch.device,
    ) -> None:
        self.model   = model.to(device)
        self.sampler = sampler
        self.cfg     = config
        self.tc      = config.training
        self.device  = device

        self.loss_fn = build_loss(
            self.tc.loss_fn,
            config.forecast_horizon,
            self.tc.loss_weights_by_horizon,
            self.tc.horizon_decay,
        )
        self.optimiser = AdamW(
            model.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay
        )

        if self.tc.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimiser, T_max=self.tc.max_epochs, eta_min=1e-6
            )
        elif self.tc.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimiser, patience=5, factor=0.5, min_lr=1e-6
            )
        else:
            self.scheduler = None

        self._best_val_loss    = math.inf
        self._patience_counter = 0
        self._checkpoint_path  = Path(self.tc.checkpoint_path)

    # ------------------------------------------------------------------

    def fit(
        self,
        station_meas: np.ndarray,
        station_nearest_grid: np.ndarray,
        grid_icond2_runs: np.ndarray,
        station_ecmwf_nwp: np.ndarray,
        station_static: np.ndarray,
        ecmwf_nwp: np.ndarray,
        icond2_static: np.ndarray,
        ecmwf_static: np.ndarray,
        train_run_pairs: list[tuple[int, int, int]],
        val_run_pairs:   list[tuple[int, int, int]],
        train_station_indices: list[int],
        val_station_indices:   list[int],
    ) -> dict:
        shared = dict(
            station_meas=station_meas,
            station_nearest_grid=station_nearest_grid,
            grid_icond2_runs=grid_icond2_runs,
            station_ecmwf_nwp=station_ecmwf_nwp,
            station_static=station_static,
            ecmwf_nwp=ecmwf_nwp,
            icond2_static=icond2_static,
            ecmwf_static=ecmwf_static,
            train_station_indices=train_station_indices,
        )

        history: dict[str, list] = {
            "epoch": [], "train_loss": [], "train_rmse": [], "train_r2": [],
            "val_loss": [], "val_rmse": [], "val_r2": [], "lr": [],
        }
        stopped_epoch = self.tc.max_epochs

        for epoch in range(1, self.tc.max_epochs + 1):
            tr_loss, tr_rmse, tr_r2 = self._train_epoch(train_run_pairs, **shared)
            va_loss, va_rmse, va_r2 = self._val_epoch(
                val_run_pairs=val_run_pairs,
                val_station_indices=val_station_indices,
                **shared,
            )

            lr = self.optimiser.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d | "
                "train loss=%.4f rmse=%.4f r²=%.4f | "
                "val   loss=%.4f rmse=%.4f r²=%.4f | "
                "lr=%.2e",
                epoch,
                tr_loss, tr_rmse, tr_r2,
                va_loss, va_rmse, va_r2,
                lr,
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["train_rmse"].append(tr_rmse)
            history["train_r2"].append(tr_r2)
            history["val_loss"].append(va_loss)
            history["val_rmse"].append(va_rmse)
            history["val_r2"].append(va_r2)
            history["lr"].append(lr)

            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(va_loss)

            if va_loss < self._best_val_loss:
                self._best_val_loss    = va_loss
                self._patience_counter = 0
                self._save_checkpoint()
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.tc.patience:
                    stopped_epoch = epoch
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.4f)",
                        epoch, self._best_val_loss,
                    )
                    break

        return {
            "history": history,
            "best_val_loss": self._best_val_loss,
            "stopped_epoch": stopped_epoch,
        }

    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_run_pairs: list[tuple[int, int, int]],
        **shared,
    ) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds:   list[Tensor] = []
        all_targets: list[Tensor] = []

        shuffled = train_run_pairs.copy()
        random.shuffle(shuffled)

        for i in range(0, len(shuffled), self.tc.batch_size):
            batch      = shuffled[i : i + self.tc.batch_size]
            batch_loss = torch.tensor(0.0, device=self.device)

            for r_curr, r_hist, t_run_abs in batch:
                sample = self.sampler.sample_train(
                    r_curr=r_curr, r_hist=r_hist, t_run_abs=t_run_abs, **shared,
                )
                loss, preds = self._forward_loss(sample, return_preds=True)
                batch_loss = batch_loss + loss
                all_preds.append(preds.detach())
                all_targets.append(sample.ground_truth.to(self.device))

            batch_loss = batch_loss / len(batch)
            self.optimiser.zero_grad()
            batch_loss.backward()
            if self.tc.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.gradient_clip)
            self.optimiser.step()
            total_loss += batch_loss.item()

        n_batches = max(len(shuffled) // self.tc.batch_size, 1)
        avg_loss  = total_loss / n_batches
        _, rmse, r2 = _metrics(
            torch.cat(all_preds), torch.cat(all_targets)
        )
        return avg_loss, rmse, r2

    @torch.no_grad()
    def _val_epoch(
        self,
        val_run_pairs: list[tuple[int, int, int]],
        val_station_indices: list[int],
        **shared,
    ) -> tuple[float, float, float]:
        self.model.eval()
        total_loss   = 0.0
        all_preds:   list[Tensor] = []
        all_targets: list[Tensor] = []

        for r_curr, r_hist, t_run_abs in val_run_pairs:
            sample = self.sampler.sample_val(
                r_curr=r_curr, r_hist=r_hist, t_run_abs=t_run_abs,
                val_station_indices=val_station_indices,
                **shared,
            )
            loss, preds = self._forward_loss(sample, return_preds=True)
            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(sample.ground_truth.to(self.device))

        avg_loss = total_loss / max(len(val_run_pairs), 1)
        _, rmse, r2 = _metrics(
            torch.cat(all_preds), torch.cat(all_targets)
        )
        return avg_loss, rmse, r2

    def _forward_loss(
        self, sample: SampleBatch, return_preds: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        data        = sample.data.to(self.device)
        target_mask = sample.target_mask.to(self.device)
        gt          = sample.ground_truth.to(self.device)

        preds = self.model(data, target_mask)

        if torch.isnan(gt).any():
            raise ValueError(f"NaN in ground truth. Shape: {gt.shape}, count: {torch.isnan(gt).sum().item()}")
        if torch.isnan(preds).any():
            # Find which input tensors contain NaN
            nan_info = []
            for key, tensor in [
                ("station.x",      data["station"].x),
                ("icond2.x",       data["icond2"].x),
                ("ecmwf.x",        data["ecmwf"].x),
                ("station.static", data["station"].static),
            ]:
                if torch.isnan(tensor).any():
                    nan_info.append(f"{key}: {torch.isnan(tensor).sum().item()} NaN")
            raise ValueError(
                f"NaN in model predictions — NaN in inputs: {nan_info or 'none found (NaN introduced in model)'}"
            )
        loss = self.loss_fn(preds, gt)

        if return_preds:
            return loss, preds.detach()
        return loss

    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "best_val_loss":   self._best_val_loss,
            },
            self._checkpoint_path,
        )
        logger.debug("Checkpoint → %s", self._checkpoint_path)

    def load_best(self) -> None:
        ckpt = torch.load(self._checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            "Loaded best checkpoint (val_loss=%.4f) from %s",
            ckpt["best_val_loss"], self._checkpoint_path,
        )