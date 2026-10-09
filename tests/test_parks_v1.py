"""parks_v1 extensions: categorical park-id static, ECMWF site_runs loader,
power_col target -- unit tests on synthetic inputs (no data on disk needed)."""

import numpy as np
import pandas as pd
import pytest
import torch

from utils import preprocessing
from utils.models import TFT, _static_cardinalities


def _config(static_features=('park_id',), categorical=('park_id',)):
    return {'data': {'files': ['SEL_B', 'SEL_A'], 'val_files': ['SEL_C', 'SEL_A']},
            'params': {'static_features': list(static_features),
                       'static_categorical': list(categorical)}}


def test_static_categories_order_is_files_then_val_files():
    assert preprocessing.static_categories(_config()) == {'park_id': ['SEL_B', 'SEL_A', 'SEL_C']}


def test_static_categories_rejects_unsupported_names():
    with pytest.raises(ValueError):
        preprocessing.static_categories(_config(categorical=('hub_height',)))


def test_static_categories_empty_without_key():
    assert preprocessing.static_categories({'data': {}, 'params': {}}) == {}


def test_cardinalities_follow_static_feature_order():
    cfg = _config(static_features=('altitude', 'park_id'))
    assert _static_cardinalities(cfg, 2) == [0, 3]
    assert _static_cardinalities({'data': {}, 'params': {'static_features': ['altitude']}}, 1) is None
    with pytest.raises(ValueError):
        _static_cardinalities(cfg, 1)


def _tft(cards):
    torch.manual_seed(0)
    return TFT(observed_dim=1, known_dim=2, static_dim=len(cards), hidden_dim=8, num_heads=2,
               lookback=4, horizon=3, static_embedding_dim=8, static_cardinalities=cards)


def test_tft_categorical_static_uses_embedding_table():
    model = _tft([3])
    assert isinstance(model.static_embed[0], torch.nn.Embedding)
    assert model.static_embed[0].num_embeddings == 3
    obs, known = torch.randn(4, 4, 1), torch.randn(4, 7, 2)
    # codes arrive as floats with round-off from the dataframe pipeline
    static = torch.tensor([[0.0], [1.0000001], [1.9999999], [2.0]])
    out = model(obs, known, static)
    assert out.shape[0] == 4 and torch.isfinite(out).all()
    model.eval()
    same = torch.tensor([[1.0], [1.0]])
    a = model(obs[:2], known[:2], same)
    b = model(obs[:2], known[:2], torch.tensor([[2.0], [2.0]]))
    assert not torch.allclose(a, b)   # different parks -> different embedding -> different output


def test_tft_numeric_static_unchanged():
    model = _tft([0])
    assert isinstance(model.static_embed[0], torch.nn.Linear)
    out = model(torch.randn(2, 4, 1), torch.randn(2, 7, 2), torch.randn(2, 1))
    assert torch.isfinite(out).all()


def test_tft_rejects_code_out_of_range():
    model = _tft([2])
    with pytest.raises(IndexError):
        model(torch.randn(1, 4, 1), torch.randn(1, 7, 2), torch.tensor([[5.0]]))


def _write_ecmwf(path, run, lat, lon, value):
    path.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp('2024-03-10', tz='UTC') + pd.Timedelta(hours=int(run))
    df = pd.DataFrame({'starttime': [start] * 3, 'forecasttime': [0, 1, 2],
                       'u_wind100m': [value] * 3, 'v_wind100m': [0.0] * 3})
    df.to_parquet(path / f"{int(lat)}_{str(lat).split('.')[1]}_{int(lon)}_{str(lon).split('.')[1]}_wind_sl.parquet")


def test_ecmwf_site_runs_picks_nearest_and_pools_runs(tmp_path):
    for run in ('00', '12'):
        site = tmp_path / 'SL' / run / 'park_X'
        _write_ecmwf(site, run, 53.25, 14.25, 1.0)    # nearest to (53.26, 14.26)
        _write_ecmwf(site, run, 53.5, 14.5, 2.0)
    df = preprocessing._fetch_ecmwf_data_from_site_runs(
        53.26, 14.26, 1, str(tmp_path), 'park_X', ['u_wind100m', 'v_wind100m'])
    assert set(df['u_wind100m']) == {1.0}
    assert sorted(df['starttime'].dt.hour.unique()) == [0, 12]
    assert (df['rank'] == 1).all() and len(df) == 6
    two = preprocessing._fetch_ecmwf_data_from_site_runs(
        53.26, 14.26, 2, str(tmp_path), 'park_X', ['u_wind100m'])
    assert two.groupby('rank')['u_wind100m'].first().tolist() == [1.0, 2.0]


def test_ecmwf_site_runs_empty_when_site_missing(tmp_path):
    (tmp_path / 'SL' / '00').mkdir(parents=True)
    assert preprocessing._fetch_ecmwf_data_from_site_runs(
        53.0, 14.0, 1, str(tmp_path), 'park_missing', ['u_wind100m']).empty


def test_park_group_statics_capacity_weighted():
    import pandas as pd
    from utils.preprocessing import park_group_statics
    g = pd.DataFrame({'n_turbines': [3, 1], 'rated_kw': [2000.0, 4000.0], 'hub_height': [100.0, 150.0],
                      'cut_in': [3.0, 2.0], 'cut_out': [25.0, 22.0], 'rated': [12.0, 10.0],
                      'commissioning_date': ['2004-01-01', '2020-01-01']})
    s = park_group_statics(g, '2024-01-01')
    # weights 6000 : 4000 = 0.6 : 0.4
    assert s['hub_height'] == pytest.approx(0.6 * 100 + 0.4 * 150)
    assert s['cut_in'] == pytest.approx(2.6) and s['cut_out'] == pytest.approx(23.8)
    assert s['rated_wind_speed'] == pytest.approx(11.2)
    assert s['park_age'] == pytest.approx(0.6 * 20.0 + 0.4 * 4.0, abs=0.02)
    assert park_group_statics(g, pd.Timestamp('2024-01-01', tz='UTC')) == s     # tz-aware reference
