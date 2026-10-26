"""Few-shot fine-tuning for train_cl.py: continue training a saved central model with its scalers.

A pretrained run (`--save_model`) leaves `models/<study>.pt` and `models/scaler/scaler_<study>.pkl`
(+ `scaler_y_<study>.pkl` for non-power targets). `train_cl.py --init-model <pt> --finetune-days N`
starts from those weights, keeps the scalers (a 1-day window cannot define the scaling) and trains
on the N days before data.test_start. The window is set on data.train_start/train_end, so the usual
split applies; everything else (val_files for logging/early stopping, holdout_files for the
evaluation) comes from the config.
"""
import logging
import os
import pickle

import pandas as pd
import torch


def window(test_start, days: int, freq: str, horizon: int, forecast_hours=None) -> tuple:
    """(train_start, train_end) for `days` fine-tuning windows per park before the test window.

    A window at run t0 needs the run t0 - H (H = horizon steps * freq) for its observed history
    (preprocessing.create_tft_sequences) and targets [t0, t0 + H). The last fine-tuning run is
    the one whose targets end at the first test run (no target overlap with the test window);
    the window starts H earlier than the first of the `days` daily target runs.
    """
    h = int(horizon) * pd.Timedelta(freq)
    t0 = pd.Timestamp(test_start)
    first_hour = min(int(x) for x in forecast_hours) if forecast_hours else 0
    first_test_run = t0.normalize() + pd.Timedelta(hours=first_hour)
    if first_test_run < t0:
        first_test_run += pd.Timedelta(days=1)
    end = first_test_run - h
    start = end - h - pd.Timedelta(days=days - 1)
    return start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')


def scaler_path(model_path: str) -> str:
    """models/<study>.pt -> models/scaler/scaler_<study>.pkl (the train_cl.py saving convention)."""
    d, name = os.path.split(model_path)
    return os.path.join(d, 'scaler', 'scaler_' + os.path.splitext(name)[0] + '.pkl')


def load_init(model_path: str, scaler_file: str = None) -> tuple:
    """(state_dict, (scaler_x, scaler_y | None)); the torch.compile prefix is stripped from the keys."""
    sd = torch.load(model_path, map_location='cpu')
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    sp = scaler_file or scaler_path(model_path)
    with open(sp, 'rb') as f:
        sx = pickle.load(f)
    d, name = os.path.split(sp)
    syp = os.path.join(d, name.replace('scaler_', 'scaler_y_', 1))
    sy = None
    if os.path.exists(syp):
        with open(syp, 'rb') as f:
            sy = pickle.load(f)
    logging.info("Init model %s (%d tensors), scaler %s%s", model_path, len(sd), sp, ', scaler_y' if sy is not None else '')
    return sd, (sx, sy)
