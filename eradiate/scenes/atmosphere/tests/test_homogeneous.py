import numpy as np
import pinttr
import pytest

import eradiate
from eradiate.radprops.rayleigh import compute_sigma_s_air
from eradiate.scenes.atmosphere._homogeneous import HomogeneousAtmosphere
from eradiate.scenes.core import KernelDict
from eradiate._util import onedict_value
from eradiate import unit_registry as ureg


@pytest.mark.parametrize("ref", (False, True))
def test_homogeneous(mode_mono, ref):
    # This test checks the functionality of HomogeneousAtmosphere

    from eradiate.kernel.core.xml import load_dict

    # Check if default constructor works
    r = HomogeneousAtmosphere()
    assert r.toa_altitude == "auto"
    assert r.kernel_offset == ureg.Quantity(0.1, "km")
    assert r.kernel_height == ureg.Quantity(100.1, "km")

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
    r = HomogeneousAtmosphere(sigma_s=1e-5)
    assert r.sigma_s.value == ureg.Quantity(1e-5, ureg.m ** -1)

    eradiate.set_mode("mono", wavelength=650.)
    r = HomogeneousAtmosphere(toa_altitude=ureg.Quantity(10, ureg.km))
    assert r.toa_altitude == ureg.Quantity(10, ureg.km)

    # Check if sigma_s was correctly computed using the mode wavelength value
    wavelength = eradiate.mode().wavelength
    assert np.isclose(r._sigma_s.value, compute_sigma_s_air(wavelength=wavelength))

    # Check if automatic scene width works as intended
    assert np.isclose(r.kernel_width, 10. / compute_sigma_s_air(wavelength=wavelength))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(r).load() is not None

    # Check that sigma_s wavelength specification is correctly taken from eradiate mode
    eradiate.set_mode("mono", wavelength=650.)
    r = HomogeneousAtmosphere(sigma_s="auto")
    assert np.allclose(r._sigma_s.value, compute_sigma_s_air(wavelength=650.))

    # Check that attributes wrong units or invalid values raise an error
    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(toa_altitude=ureg.Quantity(10, "second"))

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(width=ureg.Quantity(5, "m^2"))

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(sigma_s=ureg.Quantity(1e-7, "m"))

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(toa_altitude=-100.)

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(width=-50.)
