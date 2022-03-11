import numpy as np
import pytest
import xarray as xr

from eradiate.thermoprops.afgl_1986 import make_profile
from eradiate.thermoprops.util import (
    column_number_density,
    number_density_at_surface,
)
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize(
    "model_id",
    [
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ],
)
def test_make_profile(mode_mono, model_id):
    """
    All thermophysical properties profile can be made.
    """
    ds = make_profile(model_id=model_id)
    assert isinstance(ds, xr.Dataset)


@pytest.mark.parametrize(
    "model_id",
    [
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ],
)
@pytest.mark.parametrize(
    "levels",
    [
        ureg.Quantity(np.linspace(0, 120, 121), "km"),
        ureg.Quantity(np.linspace(0, 86, 87), "km"),
        ureg.Quantity(np.linspace(0, 40, 41), "km"),
    ],
)
def test_make_profile_levels(mode_mono, model_id, levels):
    """
    Atmospheric thermophysical profile altitude levels array has the same
    shape as the input 'levels'.
    """
    ds = make_profile(model_id=model_id, levels=levels)
    assert ds.z_level.shape == levels.shape


@pytest.mark.parametrize(
    "model_id",
    [
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ],
)
@pytest.mark.parametrize(
    "levels",
    [
        ureg.Quantity(np.linspace(0, 120, 121), "km"),
        ureg.Quantity(np.linspace(0, 86, 87), "km"),
        ureg.Quantity(np.linspace(0, 40, 41), "km"),
    ],
)
def test_make_profile_concentrations(mode_mono, model_id, levels):
    """
    Concentration-rescaled profile concentrations match the input concentrations.
    """
    concentrations = {
        "H2O": ureg.Quantity(5e23, "m^-2"),
        "O3": ureg.Quantity(0.5, "dobson_unit"),
        "CO2": ureg.Quantity(400e-6, ""),
    }
    ds = make_profile(model_id=model_id, levels=levels, concentrations=concentrations)

    column_amount_H2O = column_number_density(ds=ds, species="H2O")
    column_amount_O3 = column_number_density(ds=ds, species="O3")
    surface_amount_CO2 = ds.mr.sel(species="CO2").values[0]

    assert np.isclose(column_amount_H2O, concentrations["H2O"], rtol=1e-9)
    assert np.isclose(column_amount_O3, concentrations["O3"], rtol=1e-9)
    assert np.isclose(surface_amount_CO2, concentrations["CO2"], rtol=1e-9)


@pytest.mark.parametrize(
    "model_id",
    [
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ],
)
@pytest.mark.parametrize(
    "levels",
    [
        ureg.Quantity(np.linspace(0, 120, 121), "km"),
        ureg.Quantity(np.linspace(0, 86, 87), "km"),
        ureg.Quantity(np.linspace(0, 40, 41), "km"),
    ],
)
def test_make_profile_concentrations_invalid(mode_mono, model_id, levels):
    """
    Too large concentrations raise.
    """
    with pytest.raises(ValueError):
        make_profile(
            model_id=model_id,
            levels=levels,
            concentrations={"CO2": ureg.Quantity(400, "")},
        )
