import pytest

import eradiate
from eradiate.scenes.spectra import SolarIrradianceSpectrum, UniformSpectrum
from eradiate.util.collections import onedict_value


def test_uniform(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Check if we can instantiate the helper
    s = UniformSpectrum()
    # Check that produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None


def test_irradiance_spectrum(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Check if we can instantiate the helper
    s = SolarIrradianceSpectrum()
    # Check that produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check that a more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum({"scale": 2.0})
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Check that the helper doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum({"dataset": "thuillier_2003"})

    with pytest.raises(ValueError):
        eradiate.mode.config["wavelength"] = 2400.
        print(eradiate.mode)
        s.kernel_dict()
