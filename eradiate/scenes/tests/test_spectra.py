import numpy as np
import pytest
from pint import DimensionalityError

import eradiate
from eradiate.scenes.spectra import SolarIrradianceSpectrum, UniformSpectrum
from eradiate.util.collections import onedict_value
from eradiate.util.units import config_default_units as cdu
from eradiate.util.units import kernel_default_units as kdu


def test_uniform(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Check if we can instantiate the element
    s = UniformSpectrum()
    assert s.value == 1.
    assert s.value_units == cdu.get("radiance")

    # Check that produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check if inconsistent units are detected
    with pytest.raises(DimensionalityError):
        UniformSpectrum(value=1., value_units="W/m^2/sr/nm", quantity="irradiance")

    # Check if default units are correctly applied
    s = UniformSpectrum(value=1., quantity="irradiance")
    assert s.value_units == cdu.get("irradiance")

    # Check if unit scaling is prorperly applied
    with cdu.override({"radiance": "W/m^2/sr/nm"}):
        s = UniformSpectrum(value=1., quantity="radiance")
    with kdu.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict()
        assert np.allclose(d["spectrum"]["value"], 1e-3)


def test_irradiance_spectrum(mode_mono):
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
        eradiate.mode.config["wavelength"] = 2400.
        s.kernel_dict()
