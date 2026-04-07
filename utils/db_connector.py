"""
Database connector for ICON-D2 WeatherDB.

Provides connection pooling, query helpers, and error handling for
PostgreSQL/PostGIS database access.
"""

import os
import logging
from urllib.parse import urlparse
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# Cache: (grid_lon, grid_lat) → exact geom bytes from multilevelfields.
# Avoids repeated ST_DWithin lookups for the same grid point across HPO trials.
_geom_cache: Dict[Tuple[float, float], Any] = {}


class WeatherDBConnector:
    """Singleton connection pool manager for WeatherDB."""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeatherDBConnector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection pool from WEATHER_DB_URL environment variable."""
        if self._pool is None:
            db_url = os.environ.get('WEATHER_DB_URL')
            if not db_url:
                raise ValueError(
                    "WEATHER_DB_URL environment variable not set. "
                    "Set it in your .bashrc or shell environment."
                )
            
            url = urlparse(db_url)
            
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=url.hostname,
                    port=url.port or 5432,
                    database=url.path[1:],
                    user=url.username,
                    password=url.password
                )
                logging.info(f"WeatherDB connection pool initialized (database: {url.path[1:]})")
            except Exception as e:
                logging.error(f"Failed to create WeatherDB connection pool: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for getting a connection from the pool."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close_all(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logging.info("WeatherDB connection pool closed")


def find_nearest_grid_points(
    station_lat: float,
    station_lon: float,
    n_points: int = 1,
    method: str = 'geodesic_next'
) -> List[Tuple[float, float, Any]]:
    """
    Find nearest ICON-D2 grid points to a station location.
    
    Args:
        station_lat: Station latitude (WGS84)
        station_lon: Station longitude (WGS84)
        n_points: Number of nearest grid points to return
        method: 'geodesic_next' (distance-ranked) or 'relative_position' (compass-labeled)
    
    Returns:
        List of tuples: (grid_lon, grid_lat, label)
        - For 'geodesic_next': label is 1-based rank (1, 2, 3, ...)
        - For 'relative_position': label is compass direction ('NW', 'NE', 'SW', 'SE', 'C')
    """
    db = WeatherDBConnector()
    
    # icon_d2_grid_points stores POINT(lat, lon) — non-standard.
    # ST_X(geom) = lat, ST_Y(geom) = lon.
    # ST_MakePoint(lat, lon) matches this non-standard storage for KNN.
    # multilevelfields uses standard POINT(lon, lat) — handled separately in load_multilevel_data.
    query = """
        SELECT
            ST_Y(geom) as lon,
            ST_X(geom) as lat
        FROM icon_d2_grid_points
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        LIMIT %s;
    """

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (station_lat, station_lon, n_points))
                results = cur.fetchall()

        if not results:
            raise ValueError(f"No grid points found near ({station_lat}, {station_lon})")

        if method == 'geodesic_next':
            # Label by distance rank: 1, 2, 3, ...
            return [(lon, lat, i) for i, (lon, lat) in enumerate(results, 1)]

        elif method == 'relative_position':
            # Label by compass direction relative to station
            labeled_points = []
            for lon, lat in results:
                # Determine quadrant
                delta_lat = lat - station_lat
                delta_lon = lon - station_lon
                
                if abs(delta_lat) < 0.01 and abs(delta_lon) < 0.01:
                    label = 'C'  # Center (very close to station)
                elif delta_lat >= 0 and delta_lon >= 0:
                    label = 'NE'
                elif delta_lat >= 0 and delta_lon < 0:
                    label = 'NW'
                elif delta_lat < 0 and delta_lon >= 0:
                    label = 'SE'
                else:
                    label = 'SW'
                
                labeled_points.append((lon, lat, label))
            
            # Ensure unique labels (if duplicates, append numeric suffix)
            seen_labels = {}
            unique_labeled = []
            for lon, lat, label in labeled_points:
                if label in seen_labels:
                    seen_labels[label] += 1
                    unique_label = f"{label}{seen_labels[label]}"
                else:
                    seen_labels[label] = 1
                    unique_label = label
                unique_labeled.append((lon, lat, unique_label))
            
            return unique_labeled
        
        else:
            raise ValueError(f"Unknown method: '{method}'. Use 'geodesic_next' or 'relative_position'.")
    
    except Exception as e:
        logging.error(f"Error finding grid points: {e}")
        raise


def load_multilevel_data(
    grid_points: List[Tuple[float, float, Any]],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    height_levels: List[int],
    features: List[str] = None
) -> pd.DataFrame:
    """
    Load ICON-D2 multilevel data from database for specified grid points and time range.

    Args:
        grid_points: List of (lon, lat, label) tuples from find_nearest_grid_points()
        start_time: Start timestamp (UTC, timezone-aware)
        end_time: End timestamp (UTC, timezone-aware)
        height_levels: List of midpoint heights (e.g., [10, 38, 78, 127, 184, 247]).
                       Each value is matched via ROUND((toplevel + bottomlevel) / 2.0).
        features: List of feature names to load (default: ['u_wind', 'v_wind', 'temperature', 'pressure'])

    Returns:
        DataFrame with columns: timestamp, starttime, forecasttime,
                               {feature}_h{height}_{label} for each grid point
    """
    if features is None:
        features = ['u_wind', 'v_wind', 'temperature', 'pressure']

    db = WeatherDBConnector()

    feature_cols = ', '.join(features)
    height_midpoints = [float(h) for h in height_levels]
    col_names = ['timestamp', 'starttime', 'forecasttime', 'height'] + features

    # If geom is cached: single indexed query (fastest path).
    # If not cached: CTE resolves the exact geom from multilevelfields in one round-trip,
    # caches it, and fetches the data — all in a single query.
    cached_query_template = """
        SELECT
            starttime + (forecasttime || ' hours')::INTERVAL AS timestamp,
            starttime,
            forecasttime,
            ROUND(CAST((toplevel + bottomlevel) / 2.0 AS NUMERIC))::integer AS height,
            {feature_cols}
        FROM multilevelfields
        WHERE geom = %s
          AND starttime >= %s
          AND starttime < %s
          AND ROUND(CAST((toplevel + bottomlevel) / 2.0 AS NUMERIC)) = ANY(%s::numeric[])
        ORDER BY starttime, forecasttime, height;
    """

    uncached_query_template = """
        WITH exact_geom AS MATERIALIZED (
            SELECT geom FROM multilevelfields
            WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(%s, %s), 4326), 0.001)
              AND starttime >= %s
            LIMIT 1
        )
        SELECT
            m.starttime + (m.forecasttime || ' hours')::INTERVAL AS timestamp,
            m.starttime,
            m.forecasttime,
            ROUND(CAST((m.toplevel + m.bottomlevel) / 2.0 AS NUMERIC))::integer AS height,
            m.geom,
            {feature_cols_m}
        FROM multilevelfields m
        JOIN exact_geom USING (geom)
        WHERE m.starttime >= %s
          AND m.starttime < %s
          AND ROUND(CAST((m.toplevel + m.bottomlevel) / 2.0 AS NUMERIC)) = ANY(%s::numeric[])
        ORDER BY m.starttime, m.forecasttime, height;
    """

    all_dfs = []

    try:
        with db.get_connection() as conn:
            for grid_lon, grid_lat, label in grid_points:
                cache_key = (grid_lon, grid_lat)

                if cache_key in _geom_cache:
                    # Fast path: exact geom already known → single indexed query
                    with conn.cursor() as cur:
                        query = cached_query_template.format(feature_cols=feature_cols)
                        cur.execute(query, (_geom_cache[cache_key], start_time, end_time, height_midpoints))
                        rows = cur.fetchall()

                    if not rows:
                        logging.warning(
                            f"No data found for grid point ({grid_lat:.4f}, {grid_lon:.4f}) "
                            f"in time range {start_time} to {end_time}"
                        )
                        continue

                    df = pd.DataFrame(rows, columns=col_names)

                else:
                    # First-time path: CTE resolves geom and fetches data in one round-trip
                    feature_cols_m = ', '.join(f'm.{f}' for f in features)
                    query = uncached_query_template.format(feature_cols_m=feature_cols_m)

                    with conn.cursor() as cur:
                        cur.execute(query, (grid_lon, grid_lat, start_time, start_time, end_time, height_midpoints))
                        rows = cur.fetchall()

                    if not rows:
                        logging.warning(
                            f"No grid point or data found near ({grid_lat:.4f}, {grid_lon:.4f}) "
                            f"in time range {start_time} to {end_time}"
                        )
                        continue

                    # geom is returned right after height (index 4); extract and cache it.
                    geom_col_idx = col_names.index('height') + 1  # = 4
                    _geom_cache[cache_key] = rows[0][geom_col_idx]
                    # Drop geom column so the remaining columns match col_names exactly.
                    rows = [r[:geom_col_idx] + r[geom_col_idx + 1:] for r in rows]

                    df = pd.DataFrame(rows, columns=col_names)

                # Pivot using pandas: one column per feature×height, named {feat}_h{height}_{label}
                pivot_parts = []
                for feat in features:
                    pivot = df.pivot_table(
                        index=['timestamp', 'starttime', 'forecasttime'],
                        columns='height',
                        values=feat,
                        aggfunc='first'
                    )
                    pivot.columns = [f'{feat}_h{c}_{label}' for c in pivot.columns]
                    pivot_parts.append(pivot)

                df_pivoted = pd.concat(pivot_parts, axis=1).reset_index()
                all_dfs.append(df_pivoted)
        
        if not all_dfs:
            raise ValueError("No data loaded from database for any grid point")
        
        # Merge all grid points on timestamp, starttime, forecasttime
        df_merged = all_dfs[0]
        for df_grid in all_dfs[1:]:
            df_merged = df_merged.merge(
                df_grid,
                on=['timestamp', 'starttime', 'forecasttime'],
                how='outer'
            )
        
        # Sort by timestamp
        df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp to datetime
        df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'], utc=True)
        
        return df_merged
    
    except Exception as e:
        logging.error(f"Error loading multilevel data: {e}")
        raise


def get_latest_forecast_run() -> pd.Timestamp:
    """Get the timestamp of the most recent forecast run in the database."""
    db = WeatherDBConnector()
    
    query = "SELECT MAX(starttime) FROM multilevelfields;"
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()[0]
        
        if result is None:
            raise ValueError("No forecast data found in database")
        
        return pd.Timestamp(result, tz='UTC')
    
    except Exception as e:
        logging.error(f"Error getting latest forecast run: {e}")
        raise


def test_connection() -> bool:
    """Test database connection. Returns True if successful, False otherwise."""
    try:
        db = WeatherDBConnector()
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                return result[0] == 1
    except Exception as e:
        logging.error(f"Database connection test failed: {e}")
        return False
