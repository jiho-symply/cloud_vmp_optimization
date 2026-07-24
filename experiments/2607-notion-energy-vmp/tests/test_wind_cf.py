import numpy as np

from renewable_trace.wind_cf import wind_capacity_factor


def test_wind_capacity_factor_below_cut_in_is_zero():
    result = wind_capacity_factor(np.array([3.49]))
    assert result[0] == 0.0


def test_wind_capacity_factor_at_cut_in_is_zero():
    result = wind_capacity_factor(np.array([3.5]))
    assert result[0] == 0.0


def test_wind_capacity_factor_between_cut_in_and_rated_uses_cubic_curve_with_loss():
    speed = 8.0
    expected_raw = (speed**3 - 3.5**3) / (12.0**3 - 3.5**3)
    result = wind_capacity_factor(np.array([speed]))
    assert result[0] == expected_raw * 0.90


def test_wind_capacity_factor_at_rated_speed_is_loss_factor():
    result = wind_capacity_factor(np.array([12.0]))
    assert result[0] == 0.90


def test_wind_capacity_factor_between_rated_and_cut_out_is_loss_factor():
    result = wind_capacity_factor(np.array([20.0]))
    assert result[0] == 0.90


def test_wind_capacity_factor_above_cut_out_is_zero():
    result = wind_capacity_factor(np.array([25.01]))
    assert result[0] == 0.0
