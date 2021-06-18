import numpy as np
import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.radprops.rayleigh import compute_sigma_s_air
from eradiate.scenes.atmosphere._homogeneous import HomogeneousAtmosphere
from eradiate.scenes.core import KernelDict


@pytest.mark.parametrize("ref", (False, True))
def test_homogeneous(mode_mono, ref):
    # This test checks the functionality of HomogeneousAtmosphere

    from mitsuba.core.xml import load_dict

    # Check if default constructor works
    r = HomogeneousAtmosphere()
    assert r.toa_altitude == "auto"
    assert r.kernel_offset() == ureg.Quantity(0.1, "km")
    assert r.kernel_height() == ureg.Quantity(100.1, "km")

    # Check if default constructs can be loaded by the kernel
    ctx = KernelDictContext(ref=False)

    dict_phase = onedict_value(r.kernel_phase(ctx))
    assert load_dict(dict_phase) is not None

    dict_medium = onedict_value(r.kernel_media(ctx))
    assert load_dict(dict_medium) is not None

    dict_shape = onedict_value(r.kernel_shapes(ctx))
    assert load_dict(dict_shape) is not None

    # Check if produced scene can be instantiated
    ctx = KernelDictContext(ref=True)

    kernel_dict = KernelDict.new(r, ctx=ctx)
    assert kernel_dict.load() is not None

    # Construct with parameters
    r = HomogeneousAtmosphere(sigma_s=1e-5)
    assert r.eval_sigma_s() == ureg.Quantity(1e-5, ureg.m ** -1)

    r = HomogeneousAtmosphere(toa_altitude=ureg.Quantity(10, ureg.km))
    assert r.toa_altitude == ureg.Quantity(10.0, ureg.km)

    # Check if automatic scene width works as intended
    wavelength = ctx.spectral_ctx.wavelength
    assert np.isclose(
        r.kernel_width(ctx), 10.0 / compute_sigma_s_air(wavelength=wavelength)
    )

    # Check if produced scene can be instantiated
    assert KernelDict.new(r, ctx=ctx).load() is not None

    # Check that sigma_s wavelength specification is correctly taken from eradiate mode
    ctx.spectral_ctx.wavelength = 650.0
    r = HomogeneousAtmosphere()
    assert np.allclose(
        r.eval_sigma_s(ctx.spectral_ctx), compute_sigma_s_air(wavelength=650.0)
    )

    # Check that attributes wrong units or invalid values raise an error
    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(toa_altitude=ureg.Quantity(10, "second"))

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(width=ureg.Quantity(5, "m^2"))

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(sigma_s=ureg.Quantity(1e-7, "m"))

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(toa_altitude=-100.0)

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(width=-50.0)
