"""utils.vsn: mean variable selection weights of a TFT, one row per network input."""

import numpy as np
import torch

from utils import vsn
from utils.models import TFT


def test_mean_weights_shapes_names_and_normalisation():
    torch.manual_seed(0)
    model = TFT(observed_dim=1, known_dim=3, static_dim=2, hidden_dim=8, num_heads=2,
                lookback=6, horizon=4, static_embedding_dim=4)
    rng = np.random.default_rng(0)
    X = {'observed': rng.normal(size=(10, 6, 1)), 'known': rng.normal(size=(10, 10, 3)),
         'static': rng.normal(size=(10, 2))}
    names = {'observed': ['power'], 'known': ['ws78', 'ws127', 'ws184'], 'static': ['age', 'alt']}
    w = vsn.mean_weights(model, X, torch.device('cpu'), names, batch_size=4)
    assert list(w[w.network == 'past'].feature) == ['power', 'ws78', 'ws127', 'ws184']
    assert list(w[w.network == 'future'].feature) == ['ws78', 'ws127', 'ws184']
    assert list(w[w.network == 'static'].feature) == ['age', 'alt']
    assert np.allclose(w.groupby('network').weight.sum(), 1.0)


def test_input_names_follow_feature_matching():
    cols = ['power', 'wind_speed_h78', 'wind_speed_h127', 'ecmwf_wind_speed_h100', 'altitude']
    n = vsn.input_names(cols, {'known': ['wind_speed_h78', 'wind_speed_h127', 'ecmwf_wind_speed_h100'],
                               'observed': ['power'], 'static': ['altitude']})
    assert n == {'observed': ['power'], 'known': ['wind_speed_h78', 'wind_speed_h127', 'ecmwf_wind_speed_h100'],
                 'static': ['altitude']}
