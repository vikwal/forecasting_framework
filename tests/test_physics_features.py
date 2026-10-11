"""Physics features of the real-park path (preprocessing.add_physics_features): NWP layer winds
interpolated to the turbine groups' hub heights and the library power curves applied."""

import numpy as np
import pandas as pd
import pytest

from utils import preprocessing as pp


def _df(n=5):
    rng = np.random.default_rng(0)
    base = rng.uniform(3, 12, n)
    return pd.DataFrame({'wind_speed_h10_1': base * 0.7, 'wind_speed_h38_1': base * 0.9,
                         'wind_speed_h78_1': base, 'wind_speed_h127_1': base * 1.1,
                         'wind_speed_h78_2': base * 5.0,          # second grid point: must be ignored
                         'temperature_h78_1': 280.0})


def _curves():
    ws = np.arange(0.0, 26.0, 1.0)
    lin = np.clip((ws - 3.0) / 9.0, 0, 1) * 2000.0               # 2 MW: cut-in 3, rated 12
    lin[ws >= 25] = 0.0
    small = np.clip((ws - 3.0) / 9.0, 0, 1) * 500.0
    small[ws >= 25] = 0.0
    c = pd.DataFrame({'wind_speed': ws, 'Big 2000': lin, 'Small 500': small})
    c.loc[5, 'Small 500'] = np.nan                                  # gaps in the library are skipped
    return c


def _groups():
    return pd.DataFrame({'turbine_name': ['Big 2000', 'Small 500'], 'n_turbines': [3, 2],
                         'rated_kw': [2000.0, 500.0], 'hub_height': [78.0, 38.0]})


def test_layer_columns_take_the_nearest_point():
    layers = pp._layer_wind_columns(_df().columns)
    assert layers == {10: 'wind_speed_h10_1', 38: 'wind_speed_h38_1', 78: 'wind_speed_h78_1', 127: 'wind_speed_h127_1'}


def test_hub_wind_hits_the_layer_values_and_interpolates_in_log_height():
    df = _df()
    layers = pp._layer_wind_columns(df.columns)
    assert np.allclose(pp.hub_wind_from_layers(df, layers, 78.0), df['wind_speed_h78_1'])
    assert np.allclose(pp.hub_wind_from_layers(df, layers, 38.0), df['wind_speed_h38_1'])
    mid = pp.hub_wind_from_layers(df, layers, 100.0)
    f = np.log(100 / 78) / np.log(127 / 78)
    assert np.allclose(mid, df['wind_speed_h78_1'] * (1 - f) + df['wind_speed_h127_1'] * f)
    assert np.allclose(pp.hub_wind_from_layers(df, layers, 200.0), df['wind_speed_h127_1'])   # clipped


def test_physics_features_capacity_weighted_and_normalised():
    df = pp.add_physics_features(_df(), _groups(), _curves(), 7000.0, ['nwp_wind_hub', 'nwp_power_curve'])
    w = np.array([6000.0, 1000.0]) / 7000.0
    expect_wind = w[0] * df['wind_speed_h78_1'] + w[1] * df['wind_speed_h38_1']
    assert np.allclose(df['nwp_wind_hub'], expect_wind)
    assert df['nwp_power_curve'].between(0, 1).all()
    v = df['wind_speed_h78_1'].to_numpy()
    big = 3 * np.clip((v - 3) / 9, 0, 1) * 2000.0
    small = 2 * np.clip((df['wind_speed_h38_1'].to_numpy() - 3) / 9, 0, 1) * 500.0
    assert np.allclose(df['nwp_power_curve'], np.clip((big + small) / 7000.0, 0, 1), atol=0.02)   # grid of 1 m/s


def test_only_requested_names_are_added():
    df = pp.add_physics_features(_df(), _groups(), _curves(), 7000.0, ['nwp_power_curve'])
    assert 'nwp_power_curve' in df and 'nwp_wind_hub' not in df


def test_needs_two_layers():
    with pytest.raises(ValueError):
        pp.add_physics_features(pd.DataFrame({'wind_speed_h78_1': [5.0]}), _groups(), _curves(), 7000.0,
                                ['nwp_power_curve'])
