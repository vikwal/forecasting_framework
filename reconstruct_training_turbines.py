import pandas as pd
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

# Load config
with open('configs/config_wind_160cl.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get turbine types from config
turbine_types = config['params']['turbines']
print(f"Available turbine types: {turbine_types}")

# Load parameter files
wind_params = pd.read_csv('data/wind_parameter.csv', sep=';', dtype={'park_id': str})
turbine_params = pd.read_csv('data/turbine_parameter.csv', sep=';', dtype={'turbines': str})

# Reference date for age calculation
reference_date = datetime(2025, 8, 1)

# seed
seed = 42

# Create dictionary to store results
station_turbines = {}

# Get all station IDs from stations_master
station_ids = config['data']['files']

print(f"\nProcessing {len(station_ids)} stations...")

for station_id in station_ids:
    # Randomly select a turbine from the available types
    np.random.seed(seed)
    selected_turbine = np.random.choice(turbine_types)

    # Get turbine parameters
    turbine_data = turbine_params[turbine_params['turbine_name'] == selected_turbine]

    if len(turbine_data) == 0:
        print(f"Warning: No data found for turbine {selected_turbine}")
        continue

    turbine_data = turbine_data.iloc[0]

    # Get station data for age calculation
    station_data = wind_params[wind_params['park_id'] == station_id]

    if len(station_data) == 0:
        print(f"Warning: No data found for station {station_id}")
        continue

    station_data = station_data.iloc[0]

    # Calculate age in years (with decimals)
    commissioning_date = pd.to_datetime(station_data['commissioning_date'])
    age_years = (reference_date - commissioning_date).days / 365.25

    # Store all parameters
    station_turbines[station_id] = {
        'turbine_type': selected_turbine,
        'rotor_diameter': turbine_data['diameter'],
        'hub_height': turbine_data['hub_height'],
        'cut_in': turbine_data['cut_in'],
        'cut_out': turbine_data['cut_out'],
        'rated_wind_speed': turbine_data['rated'],
        'park_age': age_years,
        'altitude': station_data.get('altitude', np.nan) if 'altitude' in station_data.index else np.nan
    }
    seed += 1

# Convert to DataFrame
df_station_turbines = pd.DataFrame.from_dict(station_turbines, orient='index')
df_station_turbines.index.name = 'station_id'
df_station_turbines = df_station_turbines.reset_index()

# Display results
print(f"\nCreated turbine assignment for {len(df_station_turbines)} stations")
print(f"\nFirst 10 rows:")
print(df_station_turbines.head(10))

print(f"\nStatistics:")
print(df_station_turbines.describe())

print(f"\nTurbine type distribution:")
print(df_station_turbines['turbine_type'].value_counts())

# Save to CSV
output_file = 'data/station_turbine_assignments_160.csv'
df_station_turbines.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")
