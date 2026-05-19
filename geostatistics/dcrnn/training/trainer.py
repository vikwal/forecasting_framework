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
CPU sample building and graph collation is overlapped with GPU compute via
_BatchPrefetcher: background worker thread(s) collect batch_size SampleBatch
objects, call Batch.from_data_list() on CPU to create one large disconnected
graph, then .to(device) in the main thread right before use.

Multi-worker support (prefetch_workers > 1): run_pairs are distributed
round-robin across worker threads via ThreadPoolExecutor. Each worker
independently builds and collates batches from its assigned pairs, then
puts the result into a shared queue. Multiple workers can improve throughput
when sample construction is CPU-bound. Single-worker (default) is sufficient
for most cases and has zero partitioning overhead.

True mini-batching: batch_size samples are processed as one forward pass on
a single large disconnected graph. This multiplies tensor sizes by batch_size,
substantially improving GPU utilisation compared to sequential single-graph
forward passes.  ``config.training.batch_size`` is the true batch size (not
gradient accumulation steps as in older versions).
"""
from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue as _Queue
from threading import Thread
import itertools

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.data import Batch

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:
    _SummaryWriter = None  # type: ignore[assignment,misc]

from geostatistics.stgnn.training.losses import build_loss
from geostatistics.stgnn.training.sampler import SampleBatch, TrainingSampler
from ..config import DCRNNConfig
from ..model.dcrnn import DCRNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------

def _collate(samples: list[SampleBatch]):
    """
    Collate batch_size SampleBatch objects into one batched HeteroData (CPU).

    Batch.from_data_list() creates a single large disconnected graph:
      - Node features are concatenated along dim 0 per node type
      - Edge indices are offset by cumulative node counts
      - Station graphs from different samples have no cross-sample edges

    The result is mathematically identical to processing each sample
    individually, but the larger tensors achieve far better GPU utilisation.
    """
    batched_data = Batch.from_data_list([s.data for s in samples])
    batched_mask = torch.cat([s.target_mask for s in samples])
    batched_gt   = torch.cat([s.ground_truth for s in samples])
    return batched_data, batched_mask, batched_gt


class _BatchPrefetcher:
    """
    Overlaps CPU sample building and graph collation with GPU compute.

    Supports both single-threaded and multi-threaded sample collection:
    - n_workers=1 (default): Single worker thread, no GIL contention
    - n_workers>1: ThreadPoolExecutor pools work across multiple threads

    The worker(s) collect batch_size SampleBatch objects, call
    _collate() (Batch.from_data_list on CPU), then put the result into the
    queue.  The main thread calls .to(device) right before use, so a single
    PCIe transfer moves the entire batch at once.

    Parameters
    ----------
    run_pairs  : list of (r_curr, r_hist, t_run_abs) tuples
    sample_fn  : callable(*pair) -> SampleBatch  (called in worker thread(s))
    device     : target torch.device
    batch_size : number of samples to collate per forward pass
    n_ahead    : queue depth (number of pre-collated batches to buffer)
    n_workers  : number of worker threads (default: 1)
    """

    def __init__(
        self,
        run_pairs: list,
        sample_fn,
        device: torch.device,
        batch_size: int = 1,
        n_ahead: int = 2,
        n_workers: int = 1,
    ) -> None:
        self._q          = _Queue(maxsize=n_ahead)
        self._device     = device
        self._batch_size = max(batch_size, 1)
        self._n_workers  = max(n_workers, 1)
        self._run_pairs  = run_pairs

        if self._n_workers == 1:
            # Fast path: single worker thread (no partitioning overhead)
            t = Thread(
                target=self._worker,
                args=(run_pairs, sample_fn),
                daemon=True,
            )
            t.start()
        else:
            # Multi-worker: partition run_pairs and use ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=self._n_workers)
            pair_chunks = [
                list(itertools.islice(run_pairs, i, None, self._n_workers))
                for i in range(self._n_workers)
            ]
            pair_chunks = [chunk for chunk in pair_chunks if chunk]  # filter empty
            
            for chunk in pair_chunks:
                self._executor.submit(self._worker, chunk, sample_fn)

    def _worker(self, run_pairs, sample_fn) -> None:
        """Worker thread: samples and collates batches."""
        try:
            buf: list[SampleBatch] = []
            for pair in run_pairs:
                buf.append(sample_fn(*pair))
                if len(buf) == self._batch_size:
                    self._q.put(_collate(buf))
                    buf = []
            if buf:
                self._q.put(_collate(buf))   # flush last partial batch
        except Exception as exc:
            self._q.put(exc)
        finally:
            self._q.put(None)  # signal end-of-stream

    def __iter__(self):
        """Main thread: consume batches from queue."""
        finished_workers = 0
        while finished_workers < self._n_workers:
            item = self._q.get()
            if item is None:
                finished_workers += 1
                continue
            if isinstance(item, Exception):
                raise item
            data, mask, gt = item
            yield (
                data.to(self._device),
                mask.to(self._device),
                gt.to(self._device),
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
        writer=None,  # optional SummaryWriter for TensorBoard logging
        target_scale: float = 1.0,
        target_mean: float = 0.0,
    ) -> None:
        self.model   = model.to(device)
        self.sampler = sampler
        self.cfg     = config
        self.tc      = config.training
        self.device  = device
        self.tf_start = teacher_forcing_start
        self.tf_end   = teacher_forcing_end
        self.writer   = writer
        self.target_scale = target_scale
        self.target_mean  = target_mean

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

        self._best_val_rmse    = math.inf
        self._patience_counter = 0
        self._ckpt_path        = Path(self.tc.checkpoint_path)
        self._iteration_counter: int = 0   # global batch counter for inv_sigmoid schedule

    # ------------------------------------------------------------------

    def _teacher_forcing_ratio(self, epoch: int) -> float:
        """
        Teacher forcing ratio for the current epoch.

        linear (default):
            linear decay from tf_start to tf_end over max_epochs

        inv_sigmoid (paper Appendix E):
            ε_i = τ / (τ + exp(i/τ)),  i = global batch iteration counter
            Stays near 1.0 early, then falls sharply around i≈τ·ln(τ),
            then plateaus at 0.  Uses self._iteration_counter (never reset).
        """
        if self.cfg.tf_schedule == "inv_sigmoid":
            tau = self.cfg.tf_tau
            return tau / (tau + math.exp(self._iteration_counter / tau))
        # linear (default)
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
        interpol_meas: "np.ndarray | None" = None,           # (T, N) Kriging lag, pre-scaled; None = disabled
        grid_icond2_uv_runs: "np.ndarray | None" = None,    # (R, 48, N_grid, 2) raw u/v for direction_to_adj
        station_k_nearest_grid: "np.ndarray | None" = None, # (N_stations, k) k nearest grid indices for nwp_nodes=False
        verbose: bool = True,
    ) -> dict:
        """Identical signature to InductiveTrainer.fit() for drop-in use."""
        import numpy as np

        tc = self.tc

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
                interpol_meas=interpol_meas,
                grid_icond2_uv_runs=grid_icond2_uv_runs,
                station_k_nearest_grid=station_k_nearest_grid,
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
                interpol_meas=interpol_meas,
                grid_icond2_uv_runs=grid_icond2_uv_runs,
                station_k_nearest_grid=station_k_nearest_grid,
            )

        for epoch in range(1, tc.max_epochs + 1):
            # ── Training epoch ───────────────────────────────────────────
            self.model.train()
            steps_per_epoch = math.ceil(len(train_run_pairs) / tc.batch_size)
            epoch_pairs = random.choices(train_run_pairs, k=steps_per_epoch * tc.batch_size)
            t_losses, t_preds_all, t_gt_all = [], [], []
            tf_ratio = self._teacher_forcing_ratio(epoch - 1)  # updated per batch below for inv_sigmoid

            prefetcher = _BatchPrefetcher(
                epoch_pairs, _train_sample, self.device, tc.batch_size,
                n_ahead=self.cfg.n_ahead_prefetch,
                n_workers=self.cfg.prefetch_workers,
            )
            for data, mask, gt in prefetcher:
                # For inv_sigmoid: recompute per batch so _iteration_counter drives the schedule
                tf_ratio = self._teacher_forcing_ratio(epoch - 1)
                self.optimiser.zero_grad()
                preds = self.model(
                    data, mask,
                    teacher_forcing_targets=gt,
                    teacher_forcing_ratio=tf_ratio,
                )
                loss = self.loss_fn(preds, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.gradient_clip
                )
                self.optimiser.step()
                self._iteration_counter += 1

                t_losses.append(loss.item())
                t_preds_all.append(preds.detach())
                t_gt_all.append(gt)

            t_loss = float(np.mean(t_losses))
            t_preds_cat = torch.cat(t_preds_all, dim=0)
            t_gt_cat    = torch.cat(t_gt_all,    dim=0)
            _, t_rmse, t_r2 = _metrics(t_preds_cat, t_gt_cat)
            t_rmse_phys = t_rmse * self.target_scale

            # ── Validation epoch ──────────────────────────────────────────
            self.model.eval()
            v_losses, v_preds_all, v_gt_all = [], [], []

            with torch.no_grad():
                val_prefetcher = _BatchPrefetcher(
                    val_run_pairs, _val_sample, self.device, tc.batch_size,
                    n_ahead=self.cfg.n_ahead_prefetch,
                    n_workers=self.cfg.prefetch_workers,
                )
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
            v_rmse_phys = v_rmse * self.target_scale

            # ── LR scheduler ─────────────────────────────────────────────
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(v_rmse_phys)
            elif self.scheduler is not None:
                self.scheduler.step()

            lr_now = self.optimiser.param_groups[0]["lr"]

            if self.writer is not None:
                self.writer.add_scalar("train/loss",              t_loss,   epoch)
                self.writer.add_scalar("train/rmse",              t_rmse_phys, epoch)
                self.writer.add_scalar("train/r2",                t_r2,        epoch)
                self.writer.add_scalar("val/loss",                v_loss,      epoch)
                self.writer.add_scalar("val/rmse",                v_rmse_phys, epoch)
                self.writer.add_scalar("val/r2",                  v_r2,     epoch)
                self.writer.add_scalar("train/lr",                lr_now,   epoch)
                self.writer.add_scalar("train/teacher_forcing",   tf_ratio, epoch)

            if verbose:
                logger.info(
                    "Epoch %3d/%d | TF=%.2f | "
                    "train loss=%.4f RMSE=%.4f R²=%.4f | "
                    "val loss=%.4f RMSE=%.4f R²=%.4f | lr=%.2e",
                    epoch, tc.max_epochs, tf_ratio,
                    t_loss, t_rmse_phys, t_r2,
                    v_loss, v_rmse_phys, v_r2,
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
            if v_rmse_phys < self._best_val_rmse:
                self._best_val_rmse    = v_rmse_phys
                self._patience_counter = 0
                self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self._ckpt_path)
            else:
                self._patience_counter += 1
                if self._patience_counter >= tc.patience:
                    stopped_epoch = epoch
                    logger.debug(
                        "Early stopping after %d epochs without improvement.",
                        tc.patience,
                    )
                    break

        logger.debug(
            "Training complete. Best val RMSE: %.4f", self._best_val_rmse
        )
        return {
            "history":        history,
            "best_val_rmse":  self._best_val_rmse,
            "stopped_epoch":  stopped_epoch,
        }
