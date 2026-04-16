"""
DCRNNTrainer — training loop for the Seq2Seq DCRNN.

Nearly identical to the stgnn InductiveTrainer but with two differences:
  1. Teacher forcing: passes ground truth to the decoder during training.
  2. Scheduled teacher forcing ratio: linearly decayed from
     ``teacher_forcing_start`` to ``teacher_forcing_end`` over training.

Teacher forcing schedule
------------------------
  ratio(epoch) = max(end, start - (start-end) * epoch / max_epochs)

Starting at 1.0 (always teacher forcing) and decaying to 0.0 (fully
autoregressive) is a common curriculum.  The YAML parameters
``teacher_forcing_start`` / ``teacher_forcing_end`` control this.

Performance
-----------
CPU sample building is overlapped with GPU forward/backward via
_SamplePrefetcher: a background thread pre-builds the next n_ahead
samples into a queue while the GPU is busy, hiding the numpy prep latency.
Gradient accumulation (batch_size steps per optimizer.step) further
stabilises gradients without increasing per-step VRAM.
"""
from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from queue import Queue as _Queue
from threading import Thread

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from geostatistics.stgnn.training.losses import build_loss
from geostatistics.stgnn.training.sampler import SampleBatch, TrainingSampler
from ..config import DCRNNConfig
from ..model.dcrnn import DCRNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------

class _SamplePrefetcher:
    """
    Overlaps CPU sample building with GPU compute via a background thread.

    The worker pre-builds SampleBatch objects (CPU tensors) into a bounded
    queue. The main thread consumes them and calls .to(device) right before
    use — the PCIe transfer is thus pipelined with the worker's numpy work.

    Error handling: if the worker raises, the exception is re-raised in the
    main thread on the next iteration so it is not silently swallowed.

    Parameters
    ----------
    run_pairs : list of (r_curr, r_hist, t_run_abs) tuples
    sample_fn : callable(*pair) -> SampleBatch  (called in worker thread)
    device    : target torch.device
    n_ahead   : queue depth; 2 is optimal when forward ≈ sample-build time
    """

    def __init__(
        self,
        run_pairs: list,
        sample_fn,
        device: torch.device,
        n_ahead: int = 2,
    ) -> None:
        self._q      = _Queue(maxsize=n_ahead)
        self._device = device
        t = Thread(
            target=self._worker,
            args=(run_pairs, sample_fn),
            daemon=True,   # dies with the main process; no cleanup needed
        )
        t.start()

    def _worker(self, run_pairs, sample_fn) -> None:
        try:
            for pair in run_pairs:
                self._q.put(sample_fn(*pair))
        except Exception as exc:   # propagate to main thread
            self._q.put(exc)
        finally:
            self._q.put(None)      # sentinel — always sent

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            batch: SampleBatch = item
            # .to(device) here: PCIe transfer while worker builds next sample
            yield (
                batch.data.to(self._device),
                batch.target_mask.to(self._device),
                batch.ground_truth.to(self._device),
            )


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _metrics(preds: Tensor, targets: Tensor) -> tuple[float, float, float]:
    """MSE, RMSE, R² — NaN-safe."""
    valid = ~torch.isnan(targets)
    if not valid.any():
        return float("nan"), float("nan"), float("nan")
    p, t = preds[valid], targets[valid]
    mse    = ((p - t) ** 2).mean().item()
    rmse   = math.sqrt(mse)
    ss_res = ((p - t) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, rmse, r2


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DCRNNTrainer:
    """
    Training loop for the DCRNN Seq2Seq model.

    Parameters
    ----------
    model   : DCRNN instance
    sampler : TrainingSampler  (shared with stgnn — no changes needed)
    config  : DCRNNConfig
    device  : torch.device
    """

    def __init__(
        self,
        model: DCRNN,
        sampler: TrainingSampler,
        config: DCRNNConfig,
        device: torch.device,
        teacher_forcing_start: float = 1.0,
        teacher_forcing_end: float = 0.0,
    ) -> None:
        self.model   = model.to(device)
        self.sampler = sampler
        self.cfg     = config
        self.tc      = config.training
        self.device  = device
        self.tf_start = teacher_forcing_start
        self.tf_end   = teacher_forcing_end

        self.loss_fn = build_loss(
            self.tc.loss_fn,
            config.forecast_horizon,
            self.tc.loss_weights_by_horizon,
            self.tc.horizon_decay,
        )
        self.optimiser = AdamW(
            model.parameters(),
            lr=self.tc.lr,
            weight_decay=self.tc.weight_decay,
        )

        if self.tc.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimiser, T_max=self.tc.max_epochs, eta_min=1e-6,
            )
        elif self.tc.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimiser, patience=5, factor=0.5, min_lr=1e-6,
            )
        else:
            self.scheduler = None

        self._best_val_loss    = math.inf
        self._patience_counter = 0
        self._ckpt_path        = Path(self.tc.checkpoint_path)

    # ------------------------------------------------------------------

    def _teacher_forcing_ratio(self, epoch: int) -> float:
        """Linear schedule from tf_start to tf_end over max_epochs."""
        frac = epoch / max(self.tc.max_epochs - 1, 1)
        return self.tf_start + (self.tf_end - self.tf_start) * frac

    # ------------------------------------------------------------------

    def fit(
        self,
        station_meas: "np.ndarray",
        station_nearest_grid: "np.ndarray",
        grid_icond2_runs: "np.ndarray",
        station_ecmwf_nwp: "np.ndarray",
        station_static: "np.ndarray",
        ecmwf_nwp: "np.ndarray",
        icond2_static: "np.ndarray",
        ecmwf_static: "np.ndarray",
        train_run_pairs: list,
        val_run_pairs: list,
        train_station_indices: list,
        val_station_indices: list,
        verbose: bool = True,
    ) -> dict:
        """Identical signature to InductiveTrainer.fit() for drop-in use."""
        import numpy as np

        tc   = self.tc
        accum = max(tc.batch_size, 1)

        history: dict[str, list] = {
            "epoch": [], "train_loss": [], "train_rmse": [], "train_r2": [],
            "val_loss": [], "val_rmse": [], "val_r2": [], "lr": [],
        }
        stopped_epoch = tc.max_epochs

        # Bound sample functions — closures over the data arrays.
        # Created once here so the prefetcher worker can call them by name.
        def _train_sample(r_curr, r_hist, t_run_abs) -> SampleBatch:
            return self.sampler.sample_train(
                r_curr, r_hist, t_run_abs,
                station_meas, station_nearest_grid,
                grid_icond2_runs, station_ecmwf_nwp,
                station_static, ecmwf_nwp,
                icond2_static, ecmwf_static,
                train_station_indices,
            )

        def _val_sample(r_curr, r_hist, t_run_abs) -> SampleBatch:
            return self.sampler.sample_val(
                r_curr, r_hist, t_run_abs,
                station_meas, station_nearest_grid,
                grid_icond2_runs, station_ecmwf_nwp,
                station_static, ecmwf_nwp,
                icond2_static, ecmwf_static,
                train_station_indices,
                val_station_indices,
            )

        for epoch in range(1, tc.max_epochs + 1):
            tf_ratio = self._teacher_forcing_ratio(epoch - 1)

            # ── Training epoch ───────────────────────────────────────────
            self.model.train()
            random.shuffle(train_run_pairs)
            t_losses, t_preds_all, t_gt_all = [], [], []

            self.optimiser.zero_grad()
            prefetcher = _SamplePrefetcher(train_run_pairs, _train_sample, self.device)

            for step_idx, (data, mask, gt) in enumerate(prefetcher):
                preds = self.model(
                    data, mask,
                    teacher_forcing_targets=gt,
                    teacher_forcing_ratio=tf_ratio,
                )
                loss = self.loss_fn(preds, gt) / accum
                loss.backward()

                t_losses.append(loss.item() * accum)
                t_preds_all.append(preds.detach())
                t_gt_all.append(gt)

                is_last = (step_idx + 1 == len(train_run_pairs))
                if (step_idx + 1) % accum == 0 or is_last:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.gradient_clip
                    )
                    self.optimiser.step()
                    self.optimiser.zero_grad()

            t_loss = float(np.mean(t_losses))
            t_preds_cat = torch.cat(t_preds_all, dim=0)
            t_gt_cat    = torch.cat(t_gt_all,    dim=0)
            _, t_rmse, t_r2 = _metrics(t_preds_cat, t_gt_cat)

            # ── Validation epoch ──────────────────────────────────────────
            self.model.eval()
            v_losses, v_preds_all, v_gt_all = [], [], []

            with torch.no_grad():
                val_prefetcher = _SamplePrefetcher(val_run_pairs, _val_sample, self.device)
                for data, mask, gt in val_prefetcher:
                    preds = self.model(data, mask)   # autoregressive (no TF)
                    loss  = self.loss_fn(preds, gt)
                    v_losses.append(loss.item())
                    v_preds_all.append(preds)
                    v_gt_all.append(gt)

            v_loss = float(np.mean(v_losses))
            v_preds_cat = torch.cat(v_preds_all, dim=0)
            v_gt_cat    = torch.cat(v_gt_all,    dim=0)
            _, v_rmse, v_r2 = _metrics(v_preds_cat, v_gt_cat)

            # ── LR scheduler ─────────────────────────────────────────────
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(v_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            lr_now = self.optimiser.param_groups[0]["lr"]
            if verbose:
                logger.info(
                    "Epoch %3d/%d | TF=%.2f | "
                    "train loss=%.4f RMSE=%.4f R²=%.4f | "
                    "val loss=%.4f RMSE=%.4f R²=%.4f | lr=%.2e",
                    epoch, tc.max_epochs, tf_ratio,
                    t_loss, t_rmse, t_r2,
                    v_loss, v_rmse, v_r2,
                    lr_now,
                )

            history["epoch"].append(epoch)
            history["train_loss"].append(t_loss)
            history["train_rmse"].append(t_rmse)
            history["train_r2"].append(t_r2)
            history["val_loss"].append(v_loss)
            history["val_rmse"].append(v_rmse)
            history["val_r2"].append(v_r2)
            history["lr"].append(lr_now)

            # ── Early stopping + checkpoint ───────────────────────────────
            if v_loss < self._best_val_loss:
                self._best_val_loss    = v_loss
                self._patience_counter = 0
                self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self._ckpt_path)
            else:
                self._patience_counter += 1
                if self._patience_counter >= tc.patience:
                    stopped_epoch = epoch
                    logger.info(
                        "Early stopping after %d epochs without improvement.",
                        tc.patience,
                    )
                    break

        logger.info(
            "Training complete. Best val loss: %.4f", self._best_val_loss
        )
        return {
            "history":        history,
            "best_val_loss":  self._best_val_loss,
            "stopped_epoch":  stopped_epoch,
        }
