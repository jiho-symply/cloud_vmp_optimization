from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from renewable_trace.config import SolarConfig
from renewable_trace.sites import Site


def resource_to_solar_cf(frame: pd.DataFrame, site: Site, config: SolarConfig) -> pd.DataFrame:
    required = ["ghi", "dni", "dhi", "air_temperature", "wind_speed"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"solar resource frame is missing required columns: {missing}")

    timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
    ghi = pd.Series(frame["ghi"].astype(float).to_numpy(), index=timestamps)
    dni = pd.Series(frame["dni"].astype(float).to_numpy(), index=timestamps)
    dhi = pd.Series(frame["dhi"].astype(float).to_numpy(), index=timestamps)
    temp_air = pd.Series(frame["air_temperature"].astype(float).to_numpy(), index=timestamps)
    wind_speed = pd.Series(frame["wind_speed"].astype(float).to_numpy(), index=timestamps)
    solpos = pvlib.solarposition.get_solarposition(timestamps, site.lat, site.lon)
    albedo = (
        pd.Series(frame["surface_albedo"].astype(float).to_numpy(), index=timestamps).fillna(
            config.default_albedo
        )
        if "surface_albedo" in frame.columns
        else config.default_albedo
    )
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.lat,
        surface_azimuth=config.surface_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        albedo=albedo,
    )
    poa_global = irradiance["poa_global"].clip(lower=0.0)

    temperature_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_polymer"
    ]
    temp_cell = pvlib.temperature.sapm_cell(
        poa_global=poa_global,
        temp_air=temp_air,
        wind_speed=wind_speed,
        **temperature_params,
    )
    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global,
        temp_cell=temp_cell,
        pdc0=config.module_pdc0_w,
        gamma_pdc=config.gamma_pdc,
    )
    pdc_after_losses = pdc * (1.0 - config.system_losses)
    pac = pvlib.inverter.pvwatts(
        pdc=pdc_after_losses,
        pdc0=config.inverter_pdc0_w,
        eta_inv_nom=config.inverter_nominal_efficiency,
    )
    solar_cf = pd.Series(np.asarray(pac / config.ac_rated_power_w, dtype=float), index=frame.index)
    solar_cf = solar_cf.mask(lambda values: values.abs() < 1e-12, 0.0)
    solar_cf = solar_cf.clip(lower=0.0, upper=1.0)
    solar_cf = solar_cf.replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"],
            "site_id": site.site_id,
            "state": site.state,
            "lat": site.lat,
            "lon": site.lon,
            "solar_cf": solar_cf,
        }
    )
