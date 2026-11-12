"""CPU regression checks for the quantile fine-tuning comparison."""
import copy
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import pv_quantile_finetune as run
from pv_quantile_metrics import metrics, pinball

QS = [.05, .1, .25, .5, .75, .9, .95]


class Toy(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.Linear(3, 7)

    def forward(self, observed, known, static=None):
        return self.layer(self.dropout(known))


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    X = {'observed': np.zeros((n, 2, 0), np.float32),
         'known': rng.normal(size=(n, 2, 3)).astype(np.float32)}
    return X, (10 * X['known'][..., 0]).astype(np.float32)


class QuantileTests(unittest.TestCase):
    def test_union_matches_federated_with_remainders(self):
        torch.manual_seed(42)
        initial = Toy()
        sizes = [23, 8, 17]
        data = [dataset(n, i) for i, n in enumerate(sizes)]
        # A smooth loss isolates scheduling/aggregation from pinball's zero-gradient
        # cancellation, which Adam can amplify at float32 roundoff scale.
        criterion = lambda pred, target: ((pred - target[..., None]) ** 2).mean()
        mi = 3

        def experiment(central):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            norms = []
            def clip(opt, args, kwargs):
                norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), .01)))
            optimizer.register_step_pre_hook(clip)
            entries = data
            if central:
                entries = [({k: np.concatenate([x[k] for x, _ in data]) for k in data[0][0]},
                            np.concatenate([y for _, y in data]))]
            workers = [run.ScheduledClient(copy.deepcopy(initial), x, y, 11, criterion,
                       'cpu', is_tft=True, clipnorm=None, median_idx=mi, client_id=i,
                       sizes=sizes if central else None) for i, (x, y) in enumerate(entries)]
            def dispatch(method, ids, arg):
                return [workers[i].prepare_round(*arg) if method == 'prepare_round' else
                        workers[i].step(arg) if method == 'step' else
                        workers[i].round_metrics() for i in ids]
            for epoch in range(3):
                result, steps = run.fg.run_round(dispatch, list(range(len(workers))), model,
                                              optimizer, run.fg.round_seed(42, epoch))
                self.assertEqual(steps, 3)
                self.assertEqual(sum(r['n_samples'] for r in result), sum(sizes))
            self.assertTrue(all(norm > .01 for norm in norms))
            return model.state_dict(), optimizer.state_dict()

        central, central_opt = experiment(True)
        fed, fed_opt = experiment(False)
        for key in central:
            torch.testing.assert_close(central[key], fed[key], rtol=1e-5, atol=2e-7)
        for key, state in central_opt['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, fed_opt['state'][key][name], rtol=1e-5, atol=2e-7)

    def test_metrics_shape_and_pinball(self):
        y = np.array([[0., 1.], [2., 3.]])
        pred = np.repeat(y[..., None], len(QS), axis=-1) + 1
        self.assertAlmostEqual(pinball(pred, y, QS), .5)
        day = np.array([[False, True], [True, True]])
        result = metrics(pred, y, QS, day)
        self.assertEqual(result['n_day'], 3)
        self.assertEqual(result['crossing_rate'], 0)
        self.assertEqual(result['median_rmse'], 1)
        with self.assertRaises(ValueError):
            pinball(pred[..., 0], y, QS)
        crossed = pred[..., ::-1].copy()
        crossed[..., 0] += 2
        self.assertEqual(metrics(crossed, y, QS, day)['crossing_rate'], 1)

    def test_validation_disables_dropout_and_saves_trained_state(self):
        X, y = dataset(12, 9)
        cfg = {'model': {'tft': {'quantiles': QS, 'clipnorm': 7.}, 'epochs': 2,
                        'early_stopping': {'patience': 2, 'min_delta': .0001}}}
        state = Toy(dropout=.5).state_dict()
        original_predict = run.base.predict
        def checked_predict(model, X, device):
            self.assertFalse(model.training)
            a = original_predict(model, X, device)
            np.testing.assert_array_equal(a, original_predict(model, X, device))
            return a
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(run.models, 'get_model', side_effect=lambda **kw: Toy(dropout=.5)), \
             patch.object(run.base, 'predict', side_effect=checked_predict):
            model, log, trained = run.train('local_toy', [(X, y)], [{'X_test': X, 'y_test': y}],
                                          state, cfg, {}, 'cpu', 42, QS, Path(tmp))
            for name in ['local_toy_best.pt', 'local_toy_best_trained.pt']:
                self.assertTrue((Path(tmp) / name).exists())
            saved = run.fg.load_checkpoint(str(Path(tmp) / 'local_toy_best_trained.pt'))
            self.assertEqual(saved['best_epoch'], log['best_trained_epoch'])
            for key in trained:
                torch.testing.assert_close(trained[key], saved['state_dict'][key])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
