import numpy as np
import pytest

import eradiate
from eradiate.scenes.atmosphere.homogeneous import \
    RayleighHomogeneousAtmosphere
from eradiate.scenes.atmosphere.radiative_properties.rayleigh import \
    sigma_s_air
from eradiate.scenes.core import KernelDict
from eradiate.util.collections import onedict_value
from eradiate.util.units import config_default_units, ureg


@pytest.mark.parametrize("ref", (False, True))
def test_rayleigh_homogeneous(mode_mono, ref):
    # This test checks the functionality of RayleighHomogeneousAtmosphere

    from eradiate.kernel.core.xml import load_dict

    # Check if default constructor works
    r = RayleighHomogeneousAtmosphere()

    # Check if default constructs can be loaded by the kernel
    dict_phase = onedict_value(r.phase())
    assert load_dict(dict_phase) is not None

    dict_medium = onedict_value(r.media())
    assert load_dict(dict_medium) is not None

    dict_shape = onedict_value(r.shapes())
    assert load_dict(dict_shape) is not None

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(r)
    assert kernel_dict.load() is not None

    # Construct with parameters
    eradiate.mode.config["wavelength"] = 650.
    eradiate.mode.config["wavelength_units"] = config_default_units.get("wavelength")
    r = RayleighHomogeneousAtmosphere(height=ureg.Quantity(10, ureg.km))

    # check if sigma_s was correctly computed using the mode wavelength value
    wavelength = ureg.Quantity(eradiate.mode.config["wavelength"], eradiate.mode.config["wavelength_units"])
    assert np.isclose(r._sigma_s, sigma_s_air(wavelength=wavelength))

    # check if automatic scene width works as intended
    assert np.isclose(r._width, 10. / sigma_s_air(wavelength=wavelength))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(r).load() is not None

    # Check that sigma_s wavelength specification is correctly taken from eradiate mode
    eradiate.mode.config["wavelength"] = 650.
    r = RayleighHomogeneousAtmosphere(sigma_s="auto")
    assert r._sigma_s != sigma_s_air(wavelength=550.)
