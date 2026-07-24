from __future__ import annotations

import numpy as np
import pandas as pd

from renewable_trace.sites import Site


def wind_capacity_factor(
    windspeed_80m,
    cut_in_speed_mps: float = 3.5,
    rated_speed_mps: float = 12.0,
    cut_out_speed_mps: float = 25.0,
    wind_loss_factor: float = 0.90,
) -> np.ndarray:
    v = np.asarray(windspeed_80m, dtype=float)
    result = np.full(v.shape, np.nan, dtype=float)
    valid = ~np.isnan(v)

    below_cut_in = valid & (v < cut_in_speed_mps)
    at_or_above_cut_out = valid & (v > cut_out_speed_mps)
    cubic_region = valid & (v >= cut_in_speed_mps) & (v < rated_speed_mps)
    rated_region = valid & (v >= rated_speed_mps) & (v <= cut_out_speed_mps)

    result[below_cut_in | at_or_above_cut_out] = 0.0
    denominator = rated_speed_mps**3 - cut_in_speed_mps**3
    result[cubic_region] = (
        (v[cubic_region] ** 3 - cut_in_speed_mps**3) / denominator
    ) * wind_loss_factor
    result[rated_region] = wind_loss_factor
    return np.clip(result, 0.0, 1.0)


def resource_to_wind_cf(frame: pd.DataFrame, site: Site) -> pd.DataFrame:
    if "windspeed_80m" not in frame.columns:
        raise KeyError("wind resource frame must contain windspeed_80m")

    output = pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"],
            "site_id": site.site_id,
            "state": site.state,
            "lat": site.lat,
            "lon": site.lon,
            "wind_cf": wind_capacity_factor(frame["windspeed_80m"]),
        }
    )
    return output
