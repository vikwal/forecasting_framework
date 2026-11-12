"""CNNRNN: quantile output and step_output (2026-11-12); defaults must keep the previous output shape."""
import pytest
import torch

from utils.models import CNNRNN, get_model

HP = dict(n_cnn_layers=2, filters=8, kernel_size=3, units=8, n_rnn_layers=1, dropout=0.0)


def make(**kw):
    hp = dict(HP, **kw.pop('hp', {}))
    return CNNRNN(n_features=4, output_dim=96, conv_type='cnn', rnn_type='lstm', bidirectional=False,
                  hyperparameters=hp, **kw).eval()


def test_default_shape_unchanged():
    assert make()(torch.randn(2, 192, 4)).shape == (2, 96)


def test_quantiles_last_state_and_step():
    x = torch.randn(2, 192, 4)
    assert make(num_quantiles=7)(x).shape == (2, 96, 7)
    assert make(num_quantiles=7, hp=dict(step_output=True))(x).shape == (2, 96, 7)
    assert make(hp=dict(step_output=True))(x).shape == (2, 96)


def test_step_output_uses_state_of_each_step():
    """The output of step t must not depend on inputs after t (unidirectional LSTM, 'same' conv padding aside)."""
    torch.manual_seed(0)
    m = CNNRNN(n_features=4, output_dim=96, conv_type='tcn', rnn_type='lstm', bidirectional=False,
               hyperparameters=dict(HP, step_output=True), num_quantiles=3).eval()
    x = torch.randn(1, 192, 4)
    y = x.clone()
    y[:, -1] += 10.0                                     # change only the last input step
    assert torch.allclose(m(x)[:, :-1], m(y)[:, :-1])
    assert not torch.allclose(m(x)[:, -1], m(y)[:, -1])


def test_get_model_passes_quantiles():
    cfg = {'model': {'name': 'cnn-lstm', 'feature_dim': 4, 'output_dim': 96, 'tft': {'quantiles': [0.1, 0.5, 0.9]}}}
    assert get_model(cfg, dict(HP))(torch.randn(2, 192, 4)).shape == (2, 96, 3)
    cfg['model']['tft']['quantiles'] = [0.5]
    assert get_model(cfg, dict(HP))(torch.randn(2, 192, 4)).shape == (2, 96)


def test_step_output_with_attention_rejected():
    with pytest.raises(NotImplementedError):
        make(hp=dict(step_output=True, use_attention=True))
