import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.scenes.spectra import SolarIrradianceSpectrum


def test_solar(mode_mono):
    from mitsuba.core.xml import load_dict

    # We can instantiate the element
    s = SolarIrradianceSpectrum()

    # Unsupported solar spectrum keywords raise
    with pytest.raises(ValueError):
        SolarIrradianceSpectrum(dataset="doesnt_exist")

    # Produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # A more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum(scale=2.0)
    assert load_dict(onedict_value(s.kernel_dict())) is not None

    # Element doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    with pytest.raises(ValueError):
        eradiate.set_mode("mono", wavelength=2400.0)
        s.kernel_dict()

    # solid_2017_mean dataset can be used
    eradiate.set_mode("mono", wavelength=550.0)
    s = SolarIrradianceSpectrum(dataset="solid_2017_mean")
    assert load_dict(onedict_value(s.kernel_dict()))

    # values properties interpolates the irradiance as expected
    eradiate.set_mode("mono", wavelength=550.0)
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")
    assert np.allclose(s.values, ureg.Quantity(1.87938, "W/m^2/nm"))
