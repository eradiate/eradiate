import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops import ArrayRadProfile
from eradiate.spectral_index import SpectralIndex


def test_array_rad_props_profile(mode_mono):
    """
    Assigns attributes.
    """
    levels = ureg.Quantity(np.linspace(0, 100, 12), "km")
    albedo_values = ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless)
    sigma_t_values = ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1")
    p = ArrayRadProfile(
        levels=levels,
        albedo_values=albedo_values,
        sigma_t_values=sigma_t_values,
    )

    spectral_index = SpectralIndex.new()
    assert np.allclose(p.levels, levels)
    assert np.allclose(p.eval_albedo(spectral_index), albedo_values)
    assert np.allclose(p.eval_sigma_t(spectral_index), sigma_t_values)


def test_array_rad_props_profile_eval_dataset(mode_mono):
    """
    Returns a data set.
    """
    p = ArrayRadProfile(
        levels=ureg.Quantity(np.linspace(0, 100, 12), "km"),
        albedo_values=ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless),
        sigma_t_values=ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1"),
    )
    assert isinstance(p.eval_dataset(SpectralIndex.new()), xr.Dataset)


def test_array_rad_props_profile_invalid_values(mode_mono):
    """
    Mismatching shapes in albedo_values and sigma_t_values arrays raise.
    """
    with pytest.raises(ValueError):
        ArrayRadProfile(
            levels=ureg.Quantity(np.linspace(0, 100, 12), "km"),
            albedo_values=np.linspace(0.0, 1.0, 11),
            sigma_t_values=np.linspace(0.0, 1e-5, 10),
        )
