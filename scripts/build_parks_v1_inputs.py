#!/usr/bin/env python3
"""Parameter tables for the real MaStR parks (parks_v1) in the regular wind path.

preprocess_synth_wind_icond2 needs, besides the synth_<park>.parquet files of the
release, a station table (coordinates -> nearest NWP grid point) and the
wind/turbine parameter tables. They are derived from the release metadata
(parks.csv, wind_groups.csv, sites.csv, turbine_specs_overrides.csv) and the
power-curve library; nothing in the release directory is changed.

  stations_master.csv   station_id = park_id, coordinates = mean turbine
                        coordinate (same centre the NWP point extraction used)
  wind_parameter.csv    park_id, altitude, latitude, longitude, capacity_kw,
                        commissioning_date (first unit), client_id, archetype
  turbine_parameter.csv one row per turbine group (turbine = group id)

Usage: build_parks_v1_inputs.py [--release DIR] [--out data/parks_v1]
"""
import argparse
import os

import pandas as pd

RELEASE = os.path.join(os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1'), 'synthetic', 'wind', 'parks_v1')
SPECS = os.path.expanduser('~/Work/synthetic_re_data_generation/power_curves/turbine_specs.csv')  # library the synthesis used
SPEC_FIELDS = {'cut_in': 'Einschaltgeschwindigkeit', 'cut_out': 'Abschaltgeschwindigkeit',
               'rated': 'Nennwindgeschwindigkeit'}


def turbine_specs(types, overrides: pd.DataFrame) -> pd.DataFrame:
    """cut_in/cut_out/rated per library type, '-' replaced by the release overrides."""
    lib = pd.read_csv(SPECS, sep=';').drop_duplicates('Turbine').set_index('Turbine')
    ov = {(r.turbine, {'rated_ws': 'rated'}.get(r.field, r.field)): float(r.value)
          for r in overrides.itertuples()}
    rows = []
    for t in sorted(set(types)):
        row = {'turbine_name': t}
        for key, col in SPEC_FIELDS.items():
            v = ov.get((t, key), pd.to_numeric(lib.loc[t, col], errors='coerce'))
            if pd.isna(v):
                raise ValueError(f'{t}: {key} missing in library and overrides')
            row[key] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default=RELEASE)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'parks_v1'))
    a = ap.parse_args()
    parks = pd.read_csv(os.path.join(a.release, 'parks.csv'))
    groups = pd.read_csv(os.path.join(a.release, 'wind_groups.csv'))
    sites = pd.read_csv(os.path.join(a.release, 'sites.csv')).set_index('site_id')
    overrides = pd.read_csv(os.path.join(a.release, 'turbine_specs_overrides.csv'))
    os.makedirs(a.out, exist_ok=True)

    altitude = parks.park_id.map(sites['altitude']).round(1)
    stations = pd.DataFrame({'station_id': parks.park_id, 'station_height': altitude,
                             'longitude': parks.longitude, 'latitude': parks.latitude})
    stations.to_csv(os.path.join(a.out, 'stations_master.csv'), index=False)

    wind = pd.DataFrame({'park_id': parks.park_id, 'altitude': altitude,
                         'latitude': parks.latitude, 'longitude': parks.longitude,
                         'capacity_kw': parks.capacity_kw,
                         'commissioning_date': parks.commissioning_first,
                         'client_id': parks.client_id, 'archetype': parks.archetype})
    wind.to_csv(os.path.join(a.out, 'wind_parameter.csv'), sep=';', index=False)

    specs = turbine_specs(groups.turbine_type, overrides)
    turb = groups.rename(columns={'group_id': 'turbine', 'turbine_type': 'turbine_name',
                                  'hub_height_m': 'hub_height', 'rotor_diameter_m': 'diameter'})
    turb = turb.merge(specs, on='turbine_name', how='left')[
        ['park_id', 'turbine', 'turbine_name', 'n_turbines', 'hub_height', 'diameter',
         'rated_kw', 'rated_cap_kw', 'cut_in', 'cut_out', 'rated', 'commissioning_date']]
    turb.to_csv(os.path.join(a.out, 'turbine_parameter.csv'), sep=';', index=False)
    print(f'{len(stations)} parks, {len(turb)} turbine groups -> {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
