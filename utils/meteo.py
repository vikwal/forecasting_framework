import pvlib
import logging
import numpy as np
import pandas as pd

def get_total_irradiance(ghi,
                         pressure,
                         temperature,
                         latitude,
                         longitude,
                         surface_tilt,
                         surface_azimuth,
                         time_index,
                         elevation=None,
                         dhi=None,
                         dni=None):

    if dhi is None and dni is None:
        logging.error('At least one of direct (dni) or diffuse (dhi) irradiance needs to be given.')

    # get solar position
    solpos = pvlib.solarposition.get_solarposition(
            time=time_index,
            latitude=latitude,
            longitude=longitude,
            altitude=elevation,
            temperature=temperature,
            pressure=pressure,
    )
    solar_zenith = solpos['zenith']
    solar_azimuth = solpos['azimuth']

    if dni is None:
        dni = pvlib.irradiance.dni(ghi=ghi,
                                   dhi=dhi,
                                   zenith=solar_zenith)
    if dhi is None:
        solar_zenith_rad = np.deg2rad(solar_zenith)
        dhi = ghi - dni * np.cos(solar_zenith_rad)
        dhi = np.maximum(dhi, 0)

    # get total irradiance
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(time_index),
        model='haydavies',
    )
    return total_irradiance['poa_global']