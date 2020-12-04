import numpy as np
import pytest

import eradiate
from eradiate.scenes.atmosphere.homogeneous import \
    RayleighHomogeneousAtmosphere
from eradiate.scenes.atmosphere.radiative_properties.rayleigh import \
    compute_sigma_s_air
from eradiate.scenes.core import KernelDict
from eradiate.util.collections import onedict_value
from eradiate.util.exceptions import UnitsError
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
    r = RayleighHomogeneousAtmosphere(sigma_s=1e-5)
    assert r.sigma_s == ureg.Quantity(1e-5, ureg.m ** -1)

    eradiate.set_mode("mono", wavelength=650.)
    r = RayleighHomogeneousAtmosphere(height=ureg.Quantity(10, ureg.km))
    assert r.height == ureg.Quantity(10, ureg.km)

    # check if sigma_s was correctly computed using the mode wavelength value
    wavelength = eradiate.mode.wavelength
    assert np.isclose(r._sigma_s, compute_sigma_s_air(wavelength=wavelength))

    # check if automatic scene width works as intended
    assert np.isclose(r.kernel_width, 10. / compute_sigma_s_air(wavelength=wavelength))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(r).load() is not None

    # Check that sigma_s wavelength specification is correctly taken from eradiate mode
    eradiate.set_mode("mono", wavelength=650.)
    r = RayleighHomogeneousAtmosphere(sigma_s="auto")
    assert r._sigma_s != compute_sigma_s_air(wavelength=550.)

    # Check that attributes wrong units or invalid values raise an error
    with pytest.raises(UnitsError):
        RayleighHomogeneousAtmosphere(height=ureg.Quantity(10, "second"))

    with pytest.raises(UnitsError):
        RayleighHomogeneousAtmosphere(width=ureg.Quantity(5, "m^2"))

    with pytest.raises(UnitsError):
        RayleighHomogeneousAtmosphere(sigma_s=ureg.Quantity(1e-7, "m"))

    with pytest.raises(ValueError):
        RayleighHomogeneousAtmosphere(height=-100.)

    with pytest.raises(ValueError):
        RayleighHomogeneousAtmosphere(width=-50.)

    with pytest.raises(ValueError):
        RayleighHomogeneousAtmosphere(sigma_s=-1e-7)
