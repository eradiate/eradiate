import numpy as np
import pytest

import eradiate
from eradiate.scenes.spectra import \
    SolarIrradianceSpectrum, Spectrum, \
    UniformIrradianceSpectrum, UniformRadianceSpectrum
from eradiate.util.collections import onedict_value
from eradiate.util.exceptions import UnitsError
from eradiate.util.units import config_default_units as cdu
from eradiate.util.units import kernel_default_units as kdu
from eradiate.util.units import ureg


def test_converter(mode_mono):
    # Check if dicts are correctly processed
    s = Spectrum.converter("radiance")({"type": "uniform"})
    assert s == UniformRadianceSpectrum(value=1.0)
    s = Spectrum.converter("irradiance")({"type": "uniform"})
    assert s == UniformIrradianceSpectrum(value=1.0)

    # Check if floats and quantities are correctly processed
    s = Spectrum.converter("radiance")(1.0)
    assert s == UniformRadianceSpectrum(value=1.0)
    s = Spectrum.converter("radiance")(ureg.Quantity(1e6, "W/km^2/sr/nm"))
    assert s == UniformRadianceSpectrum(value=1.0)
    with pytest.raises(UnitsError):
        s = Spectrum.converter("irradiance")(ureg.Quantity(1, "W/m^2/sr/nm"))


def test_uniform(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Check if we can instantiate the element
    s = UniformRadianceSpectrum()
    assert s.value == ureg.Quantity(1., "W/m^2/sr/nm")

    # Check that produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check if inconsistent units are detected
    with pytest.raises(UnitsError):
        s.value = ureg.Quantity(1., ureg.m)

    # Check if default units are correctly applied
    with cdu.override({"radiance": "W/km^2/sr/nm"}):
        s = UniformRadianceSpectrum(value=1.)
    assert s.value == ureg.Quantity(1., "W/km^2/sr/nm")

    # Check if unit scaling is prorperly applied
    with cdu.override({"radiance": "W/m^2/sr/nm"}):
        s = UniformRadianceSpectrum(value=1.)
    with kdu.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict()
        assert np.allclose(d["spectrum"]["value"], 1e-3)

    # Check if unit-enabled fields are correctly listed
    assert UniformRadianceSpectrum._fields_with_units() == {"value": ureg.Unit("W/m^2/sr/nm")}


def test_solar(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Check if we can instantiate the element
    s = SolarIrradianceSpectrum()

    # Check that unsupported solar spectrum keywords raise
    with pytest.raises(ValueError):
        SolarIrradianceSpectrum(dataset="doesnt_exist")

    # Check that produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check that a more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum(scale=2.0)
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check that the element doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    with pytest.raises(ValueError):
        eradiate.set_mode("mono", wavelength=2400.)
        s.kernel_dict()
