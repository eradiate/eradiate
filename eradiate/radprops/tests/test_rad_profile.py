import numpy as np
import pytest

import eradiate
from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.radprops import (
    AFGL1986RadProfile,
    ArrayRadProfile,
    RadProfileFactory,
    US76ApproxRadProfile,
)
from eradiate.thermoprops.util import (
    compute_column_number_density,
    compute_number_density_at_surface,
)


def test_rad_props_profile_factory(mode_mono):
    p = RadProfileFactory.create(
        {
            "type": "array",
            "levels": [0, 1, 2, 3],
            "albedo_values": [[[0, 1, 2, 3]]],
            "sigma_t_values": [[[0, 1, 2, 3]]],
        }
    )
    assert p is not None


def test_array_rad_props_profile(mode_mono):
    levels = ureg.Quantity(np.linspace(0, 100, 12), "km")
    albedo_values = ureg.Quantity(np.linspace(0.0, 1.0, 11), ureg.dimensionless)
    sigma_t_values = ureg.Quantity(np.linspace(0.0, 1e-5, 11), "m^-1")
    p = ArrayRadProfile(
        levels=levels,
        albedo_values=albedo_values.reshape(1, 1, len(levels) - 1),
        sigma_t_values=sigma_t_values.reshape(1, 1, len(levels) - 1),
    )
    assert isinstance(p.levels, ureg.Quantity)
    assert isinstance(p.sigma_a, ureg.Quantity)
    assert isinstance(p.sigma_s, ureg.Quantity)
    assert np.allclose(p.levels, levels)
    assert np.allclose(p.albedo, albedo_values)
    assert np.allclose(p.sigma_t, sigma_t_values)

    # to_dataset method does not fail
    assert p.to_dataset()

    # mismatching shapes in albedo_values and sigma_t_values arrays raise
    with pytest.raises(ValueError):
        ArrayRadProfile(
            levels=levels,
            albedo_values=np.linspace(0.0, 1.0, 11).reshape(1, 1, 11),
            sigma_t_values=np.linspace(0.0, 1e-5, 10).reshape(1, 1, 10),
        )


def test_us76_approx_rad_profile(mode_mono):
    # Default constructor with test absorption data set
    test_absorption_data_set = path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    p = US76ApproxRadProfile(
        absorption_data_set=test_absorption_data_set
    )

    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 86)

    # Custom altitude levels
    p = US76ApproxRadProfile(
        levels=ureg.Quantity(np.linspace(0, 120, 121), "km"),
        absorption_data_set=test_absorption_data_set,
    )
    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 120)


@pytest.mark.slow
def test_afgl1986_rad_profile(mode_mono):
    test_absorption_data_sets = {
        "CH4": path_resolver.resolve(
            "tests/spectra/absorption/CH4-spectra-4000_11502.nc"
        ),
        "CO2": path_resolver.resolve(
            "tests/spectra/absorption/CO2-spectra-4000_14076.nc"
        ),
        "CO": path_resolver.resolve(
            "tests/spectra/absorption/CO-spectra-4000_14478.nc"
        ),
        "H2O": path_resolver.resolve(
            "tests/spectra/absorption/H2O-spectra-4000_25711.nc"
        ),
        "N2O": path_resolver.resolve(
            "tests/spectra/absorption/N2O-spectra-4000_10364.nc"
        ),
        "O2": path_resolver.resolve(
            "tests/spectra/absorption/O2-spectra-4000_17273.nc"
        ),
        "O3": path_resolver.resolve("tests/spectra/absorption/O3-spectra-4000_6997.nc"),
    }

    # Default constructor with test absorption data sets
    eradiate.set_mode(
        "mono", wavelength=1500.0
    )  # in the infrared, all absorption data sets are opened
    p = AFGL1986RadProfile(
        absorption_data_sets=test_absorption_data_sets,
    )
    for x in [p.sigma_a, p.sigma_s, p.sigma_t, p.albedo]:
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (1, 1, 120)

    # Custom level altitudes (in the visible, only the H2O data set is opened)
    eradiate.set_mode("mono", wavelength=550.0)
    p = AFGL1986RadProfile(
        levels=ureg.Quantity(np.linspace(0, 100, 101), "km"),
        absorption_data_sets=test_absorption_data_sets,
    )

    # Custom concentrations
    concentrations = {
        "H2O": ureg.Quantity(5e23, "m^-2"),  # column number density in S.I. units
        "O3": ureg.Quantity(
            0.5, "dobson_unit"
        ),  # column number density in exotic units
        "CH4": ureg.Quantity(4e19, "m^-3"),  # number density at the surface
        "CO2": ureg.Quantity(400e-6, ""),  # mixing ratio at the surface
    }
    p = AFGL1986RadProfile(
        concentrations=concentrations,
        absorption_data_sets=test_absorption_data_sets,
    )

    thermoprops = p._thermoprops
    column_amount_H2O = compute_column_number_density(thermoprops, "H2O")
    column_amount_O3 = compute_column_number_density(thermoprops, "O3")
    surface_amount_CH4 = compute_number_density_at_surface(thermoprops, "CH4")
    surface_amount_CO2 = thermoprops.mr.sel(species="CO2").values[0]

    assert np.isclose(column_amount_H2O, concentrations["H2O"], rtol=1e-9)
    assert np.isclose(column_amount_O3, concentrations["O3"], rtol=1e-9)
    assert np.isclose(surface_amount_CO2, concentrations["CO2"], rtol=1e-9)
    assert np.isclose(surface_amount_CH4, concentrations["CH4"], rtol=1e-9)

    # Too large concentrations raise
    with pytest.raises(ValueError):
        AFGL1986RadProfile(
            concentrations={"CO2": ureg.Quantity(400, "")},
            absorption_data_sets=test_absorption_data_sets,
        )
