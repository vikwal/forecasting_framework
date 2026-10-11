#!/usr/bin/env python3
"""Distance to the coast of the parks_v1 park centres -> column dist_coast_km in wind_parameter.csv.

Park centre = latitude/longitude of wind_parameter.csv (mean turbine coordinate, build_parks_v1_inputs.py).
Coastline: Natural Earth 10m, clipped to North/Baltic Sea surroundings, distance in EPSG:3035 (LAEA
Europe, metric) to the nearest coastline segment; same method as the generator's
scripts/era5_bc/wp0_distance_to_coast.py. Usable as a static feature: params.static_features
[..., dist_coast_km] (power_col mode takes extra wind_parameter.csv columns by name).

Needs geopandas (not in frcst), e.g. with the generator venv:
    ~/Work/synthetic_re_data_generation/synthre/bin/python scripts/parks_v1_dist_coast.py \
        --coastline ~/Work/synthetic_re_data_generation/data/era5_bc/ne_coastline/ne_10m_coastline.shp
"""

import argparse
import os

import geopandas as gpd
import pandas as pd

BBOX = (-5.0, 45.0, 25.0, 62.0)          # lon_min, lat_min, lon_max, lat_max
CRS_METRIC = 'EPSG:3035'


def dist_coast_km(lat, lon, coastline: str) -> pd.Series:
    coast = gpd.read_file(coastline).cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]].to_crs(CRS_METRIC)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs='EPSG:4326').to_crs(CRS_METRIC)
    return pd.Series(pts.distance(coast.geometry.union_all()).to_numpy() / 1000.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--coastline', required=True, help='ne_10m_coastline.shp')
    ap.add_argument('--table', default='data/parks_v1/wind_parameter.csv')
    a = ap.parse_args()
    wind = pd.read_csv(a.table, sep=';', dtype={'park_id': str})
    wind['dist_coast_km'] = dist_coast_km(wind.latitude, wind.longitude, os.path.expanduser(a.coastline)).round(2).values
    wind.to_csv(a.table, sep=';', index=False)
    d = wind.dist_coast_km
    print(f'{a.table}: {len(wind)} parks, dist_coast_km min {d.min():.1f} / median {d.median():.1f} / max {d.max():.1f}')
    print(wind.groupby('client_id').dist_coast_km.agg(['min', 'median', 'max']).round(1).to_string())


if __name__ == '__main__':
    main()
