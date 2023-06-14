import mitsuba as mi
import numpy as np
import pint
import pytest
from pinttr.exceptions import UnitsError

import eradiate
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import InterpolatedSpectrum, spectrum_factory
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import PhysicalQuantity


@pytest.mark.parametrize(
    "tested, expected, units",
    [
        ({}, TypeError, None),
        (
            {"wavelengths": [500.0, 600.0]},
            TypeError,
            None,
        ),
        (
            {"values": [0.0, 1.0]},
            TypeError,
            None,
        ),
        (
            {"wavelengths": [500.0, 600.0], "values": [0.0, 1.0, 2.0]},
            ValueError,
            None,
        ),
        (
            {"wavelengths": [500.0, 600.0], "values": [0.0, 1.0]},
            InterpolatedSpectrum(
                wavelengths=[500.0, 600.0] * ureg.nm,
                values=[0.0, 1.0],
            ),
            None,
        ),
        (
            {
                "quantity": "irradiance",
                "wavelengths": [500.0, 600.0],
                "values": [0.0, 1.0],
            },
            InterpolatedSpectrum(
                quantity=PhysicalQuantity.IRRADIANCE,
                wavelengths=[500.0, 600.0] * ureg.nm,
                values=[0.0, 1.0] * ureg("W/m^2/nm"),
            ),
            "W/m^2/nm",
        ),
        (
            {
                "quantity": "irradiance",
                "wavelengths": [500.0, 600.0],
                "values": [0.0, 1.0] * ureg.kg,
            },
            UnitsError,
            None,
        ),
        (
            {
                "quantity": "irradiance",
                "wavelengths": [500.0, 600.0] * ureg.s,
                "values": [0.0, 1.0],
            },
            UnitsError,
            None,
        ),
        (
            {
                "wavelengths": [500.0, 600.0],
                "values": [0.0, 1.0] * ureg.dimensionless,
            },
            InterpolatedSpectrum(
                quantity=None,
                wavelengths=[500.0, 600.0] * ureg.nm,
                values=[0.0, 1.0] * ureg.dimensionless,
            ),
            None,
        ),
    ],
    ids=[
        "no_args",
        "missing_args_1",
        "missing_args_2",
        "array_shape_mismatch",
        "no_quantity",
        "no_units",
        "inconsistent_units_1",
        "inconsistent_units_2",
        "quantity_values_but_no_quantity",
    ],
)
def test_interpolated_construct(modes_all, tested, expected, units):
    if isinstance(expected, InterpolatedSpectrum):
        s = InterpolatedSpectrum(**tested)

        assert s.quantity is expected.quantity
        assert np.all(s.values == expected.values)

        if isinstance(s.values, pint.Quantity):
            assert isinstance(s.values.magnitude, np.ndarray)
        else:
            assert isinstance(s.values, np.ndarray)

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            InterpolatedSpectrum(**tested)

    else:
        raise RuntimeError


@pytest.mark.parametrize("quantity", ["dimensionless", None])
@pytest.mark.parametrize(
    "wmin, wmax, expected, isclose_kwargs",
    [
        # Easy case: integrate over full interval
        (500.0 * ureg.nm, 600.0 * ureg.nm, 50.0 * ureg.nm, {"rtol": 1e-10}),
        # Min or max falls in-between two coordinate values
        (540.0 * ureg.nm, 600.0 * ureg.nm, 42.0 * ureg.nm, {"rtol": 1e-10}),
        (550.0 * ureg.nm, 590.0 * ureg.nm, 28.0 * ureg.nm, {"rtol": 1e-10}),
        (540.0 * ureg.nm, 590.0 * ureg.nm, 32.5 * ureg.nm, {"rtol": 1e-10}),
        (530.0 * ureg.nm, 540.0 * ureg.nm, 3.5 * ureg.nm, {"rtol": 1e-10}),
        # Integrating on an interval not intersecting the support yields 0
        (400.0 * ureg.nm, 450.0 * ureg.nm, 0.0 * ureg.nm, {"atol": 1e-10}),
        (400.0 * ureg.nm, 500.0 * ureg.nm, 0.0 * ureg.nm, {"atol": 1e-10}),
        (650.0 * ureg.nm, 700.0 * ureg.nm, 0.0 * ureg.nm, {"atol": 1e-10}),
        (600.0 * ureg.nm, 700.0 * ureg.nm, 0.0 * ureg.nm, {"atol": 1e-10}),
        # Integrating on an interval covering the whole support yields correct
        # integral values
        (450.0 * ureg.nm, 650.0 * ureg.nm, 50.0 * ureg.nm, {"rtol": 1e-10}),
        (500.0 * ureg.nm, 650.0 * ureg.nm, 50.0 * ureg.nm, {"rtol": 1e-10}),
        (450.0 * ureg.nm, 600.0 * ureg.nm, 50.0 * ureg.nm, {"rtol": 1e-10}),
    ],
)
def test_interpolated_integral(
    mode_mono, quantity, wmin, wmax, expected, isclose_kwargs
):
    s = InterpolatedSpectrum(
        wavelengths=[500.0, 525.0, 550.0, 575.0, 600.0],
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
        quantity=quantity,
    )

    integral = s.integral(wmin, wmax)

    np.testing.assert_allclose(
        integral.m_as(ureg.nm), expected.m_as(ureg.nm), **isclose_kwargs
    )


@pytest.mark.parametrize(
    "quantity, values, w, expected",
    [
        (
            "dimensionless",
            [0.0, 1.0],
            [450.0, 500.0, 550.0, 600.0, 650.0] * ureg.nm,
            [0.0, 0.0, 0.5, 1.0, 0.0],
        ),
        (
            "collision_coefficient",
            [0.0, 1.0],
            [450.0, 500.0, 550.0, 600.0, 650.0] * ureg.nm,
            [0.0, 0.0, 0.5, 1.0, 0.0] * ureg.m**-1,
        ),
    ],
)
def test_interpolated_eval_mono(mode_mono, quantity, values, w, expected):
    # No quantity, unitless value
    eval = InterpolatedSpectrum(
        quantity=quantity, values=values, wavelengths=[500.0, 600.0]
    ).eval_mono(w)
    assert np.all(expected == eval)
    assert isinstance(eval, pint.Quantity)


def test_interpolated_eval_mono_wavelengths_decreasing(mode_mono):
    # decreasing wavelength values
    with pytest.raises(ValueError):
        InterpolatedSpectrum(
            quantity="dimensionless",
            values=[0.0, 1.0],
            wavelengths=[600.0, 500.0],
        )


def test_interpolated_eval(modes_all):
    if eradiate.mode().is_mono:
        si = SpectralIndex.new(w=550.0 * ureg.nm)
        expected = 0.5

    elif eradiate.mode().is_ckd:
        si = SpectralIndex.new(w=550.0 * ureg.nm, g=0)
        expected = 0.5

    else:
        raise NotImplementedError

    # Spectrum performs linear interpolation and yields units consistent with
    # quantity
    spectrum = InterpolatedSpectrum(
        quantity="dimensionless",
        wavelengths=[500.0, 600.0],
        values=[0.0, 1.0],
    )
    assert spectrum.eval(si) == expected * spectrum.values.units

    spectrum = InterpolatedSpectrum(
        quantity="irradiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
    )
    assert spectrum.eval(si) == expected * spectrum.values.units

    # If no quantity is specified, the evaluation routine returns a
    # unitless value
    spectrum = InterpolatedSpectrum(
        quantity=None,
        wavelengths=[500.0, 600.0],
        values=[0.0, 1.0],
    )
    actual = spectrum.eval(si)
    assert actual == expected
    assert not isinstance(actual, pint.Quantity)


def test_interpolated_kernel_dict(modes_all_mono):
    # Instantiate from factory
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        spectrum = spectrum_factory.convert(
            {
                "type": "interpolated",
                "quantity": "radiance",
                "wavelengths": [0.5, 0.6],
                "wavelengths_units": "micron",
                "values": [0.0, 1.0],
            }
        )

    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        # Produced kernel dict and params are valid
        mi_wrapper = check_scene_element(spectrum, mi.Texture)
        # Unit scaling is properly applied
        assert np.isclose(mi_wrapper.parameters["value"], 5e-4)


def test_interpolated_from_dataarray(mode_mono):
    da = eradiate.data.load_dataset(
        "spectra/reflectance/lambertian_soil.nc"
    ).reflectance.sel(brightness="darkest")
    spectrum = InterpolatedSpectrum.from_dataarray(quantity="reflectance", dataarray=da)
    assert np.all(spectrum.wavelengths.m_as(da.w.attrs["units"]) == da.w.values)
    assert np.all(spectrum.values.m_as(da.attrs["units"]) == da.values)
