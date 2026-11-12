"""Quantile losses and interval diagnostics for PV forecasts."""
import numpy as np


def pinball(pred, target, quantiles):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape + (len(quantiles),):
        raise ValueError('Quantile prediction shape mismatch')
    err = target[..., None] - pred
    q = np.asarray(quantiles)
    return float(np.maximum(q * err, (q - 1) * err).mean())


def metrics(pred, target, quantiles, daylight):
    """Raw pinball loss; sorted/nonnegative quantiles for valid PV intervals."""
    p, y = pred[daylight], target[daylight]
    if not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid evaluation data')
    ordered = np.maximum(np.sort(p, axis=-1), 0)
    med = ordered[:, quantiles.index(.5)]
    out = dict(n_day=len(y), pinball_day=pinball(p, y, quantiles),
               pinball_all=pinball(pred, target, quantiles),
               pinball_day_sorted=pinball(ordered, y, quantiles),
               crossing_rate=float(np.any(np.diff(p, axis=-1) < 0, axis=-1).mean()),
               median_rmse=float(np.sqrt(np.mean((med-y)**2))),
               median_r2=float(1-np.sum((med-y)**2)/np.sum((y-y.mean())**2)))
    for level, loq, hiq in [(90,.05,.95),(80,.1,.9),(50,.25,.75)]:
        lo, hi = ordered[:,quantiles.index(loq)], ordered[:,quantiles.index(hiq)]
        alpha = 1-level/100
        width = hi-lo
        score = width + 2/alpha*np.maximum(lo-y,0) + 2/alpha*np.maximum(y-hi,0)
        out.update({f'coverage_{level}': float(((y>=lo)&(y<=hi)).mean()),
                    f'width_{level}': float(width.mean()), f'interval_score_{level}':float(score.mean())})
    return out
