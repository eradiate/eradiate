import numpy as np
import pinttr
import pytest

import eradiate
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate.ckd import BinSet
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.spectra import spectrum_factory
from eradiate.scenes.spectra._interpolated import InterpolatedSpectrum


def test_interpolated_construct(modes_all):
    # Instantiating without argument fails
    with pytest.raises(TypeError):
        InterpolatedSpectrum()
    # Instantiating with missing argument fails
    with pytest.raises(TypeError):
        InterpolatedSpectrum(wavelengths=[500.0, 600.0])
    with pytest.raises(TypeError):
        InterpolatedSpectrum(values=[0.0, 1.0])
    # Shape mismatch raises
    with pytest.raises(ValueError):
        InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0, 2.0])

    # Instantiating with no quantity applies no units
    spectrum = InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0])
    assert spectrum.quantity is None
    assert isinstance(spectrum.values, np.ndarray)
    # Instantiating with a quantity applies units
    spectrum = InterpolatedSpectrum(
        quantity="irradiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
    )
    assert spectrum.values.is_compatible_with("W/m^2/nm")

    # Inconsistent units raise
    with pytest.raises(pinttr.exceptions.UnitsError):
        InterpolatedSpectrum(
            quantity="irradiance",
            wavelengths=[500.0, 600.0],
            values=[0.0, 1.0] * ureg("W/m^2"),
        )
    with pytest.raises(pinttr.exceptions.UnitsError):
        InterpolatedSpectrum(
            quantity="irradiance",
            wavelengths=[500.0, 600.0] * ureg.s,
            values=[0.0, 1.0],
        )

    # Instantiate from factory
    assert spectrum_factory.convert(
        {
            "type": "interpolated",
            "quantity": "irradiance",
            "wavelengths": [0.5, 0.6],
            "wavelengths_units": "micron",
            "values": [0.5, 1.0],
            "values_units": "kW/m^2/nm",
        }
    )


def test_interpolated_integral(mode_mono):
    s = InterpolatedSpectrum(
        wavelengths=[500.0, 525.0, 550.0, 575.0, 600.0],
        values=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    # Easy case: integrate over full interval
    assert np.isclose(
        50.0 * ureg.nm, s.integral(500.0 * ureg.nm, 600.0 * ureg.nm), rtol=1e-10
    )

    # Min or max falls in-between two coordinate values
    assert np.isclose(
        42.0 * ureg.nm, s.integral(540.0 * ureg.nm, 600.0 * ureg.nm), rtol=1e-10
    )
    assert np.isclose(
        28.0 * ureg.nm, s.integral(550.0 * ureg.nm, 590.0 * ureg.nm), rtol=1e-10
    )
    assert np.isclose(
        32.5 * ureg.nm, s.integral(540.0 * ureg.nm, 590.0 * ureg.nm), rtol=1e-10
    )
    assert np.isclose(
        3.5 * ureg.nm, s.integral(530.0 * ureg.nm, 540.0 * ureg.nm), rtol=1e-10
    )

    # Integrating on an interval not intersecting the support yields 0
    assert np.isclose(
        0.0 * ureg.nm, s.integral(400.0 * ureg.nm, 450.0 * ureg.nm), atol=1e-10
    )
    assert np.isclose(
        0.0 * ureg.nm, s.integral(400.0 * ureg.nm, 500.0 * ureg.nm), atol=1e-10
    )
    assert np.isclose(
        0.0 * ureg.nm, s.integral(650.0 * ureg.nm, 700.0 * ureg.nm), atol=1e-10
    )
    assert np.isclose(
        0.0 * ureg.nm, s.integral(600.0 * ureg.nm, 700.0 * ureg.nm), atol=1e-10
    )

    # Integrating on an interval covering the whole support yields correct
    # integral values
    assert np.isclose(
        50.0 * ureg.nm, s.integral(450.0 * ureg.nm, 650.0 * ureg.nm), rtol=1e-10
    )
    assert np.isclose(
        50.0 * ureg.nm, s.integral(500.0 * ureg.nm, 650.0 * ureg.nm), rtol=1e-10
    )
    assert np.isclose(
        50.0 * ureg.nm, s.integral(450.0 * ureg.nm, 600.0 * ureg.nm), rtol=1e-10
    )


def test_interpolated_eval(modes_all):
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_ctx = SpectralContext.new(wavelength=550.0)
        expected = 0.5

    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        bin = BinSet.from_db("10nm_test").select_bins("550")[0]
        spectral_ctx = SpectralContext.new(bindex=bin.bindexes[0])
        expected = 0.5

    else:
        assert False

    # Spectrum without quantity performs linear interpolation and yields units
    # consistent with values
    spectrum = InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0])
    assert spectrum.eval(spectral_ctx) == expected
    spectrum.values *= ureg("W/m^2/nm")
    assert spectrum.eval(spectral_ctx) == expected * ureg("W/m^2/nm")

    # Spectrum with quantity performs linear interpolation and yields units
    # consistent with quantity
    spectrum = InterpolatedSpectrum(
        quantity="irradiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
    )
    # Interpolation returns quantity
    assert spectrum.eval(spectral_ctx) == expected * ucc.get("irradiance")


def test_interpolated_kernel_dict(modes_all_mono):
    from mitsuba.core.xml import load_dict

    ctx = KernelDictContext(spectral_ctx=SpectralContext.new(wavelength=550.0))

    spectrum = InterpolatedSpectrum(
        id="spectrum",
        quantity="irradiance",
        wavelengths=[500.0, 600.0],
        values=[0.0, 1.0],
    )

    # Produced kernel dict is valid
    assert load_dict(spectrum.kernel_dict(ctx)["spectrum"]) is not None

    # Unit scaling is properly applied
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        s = InterpolatedSpectrum(
            quantity="radiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
        )
    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict(ctx)
        assert np.allclose(d["spectrum"]["value"], 5e-4)
