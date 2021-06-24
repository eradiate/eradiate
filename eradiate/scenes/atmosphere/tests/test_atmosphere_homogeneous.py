import numpy as np
import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.radprops.rayleigh import compute_sigma_s_air
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import PhaseFunctionFactory


def test_atmosphere_homogeneous_construct(mode_mono):
    # Constructing with defaults succeeds
    r = HomogeneousAtmosphere()
    assert r.toa_altitude == "auto"
    assert r.kernel_offset() == 0.1 * ureg.km
    assert r.kernel_height() == 100.1 * ureg.km

    # Constructing with custom extinction coefficient value succeeds
    r = HomogeneousAtmosphere(sigma_s=1e-5)
    assert r.eval_sigma_s() == ureg.Quantity(1e-5, ureg.m ** -1)

    # Constructing with custom TOA altitude value succeeds
    r = HomogeneousAtmosphere(toa_altitude=10.0 * ureg.km)
    assert r.toa_altitude == 10.0 * ureg.km

    #  Wrong attribute units or invalid values raise an error
    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(toa_altitude=10 * ureg.s)

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(width=5 * ureg.m ** 2)

    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(sigma_s=1e-7 * ureg.m)

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(toa_altitude=-100.0)

    with pytest.raises(ValueError):
        HomogeneousAtmosphere(width=-50.0)


@pytest.mark.parametrize("phase_id", PhaseFunctionFactory.registry.keys())
@pytest.mark.parametrize("ref", (False, True))
def test_atmosphere_homogeneous_phase_function(mode_mono, phase_id, ref):
    # All available phase functions can be used to create an instance
    r = HomogeneousAtmosphere(phase={"type": phase_id})

    # The resulting object produces a valid kernel dictionary
    ctx = KernelDictContext(ref=ref)
    kernel_dict = KernelDict.new(r, ctx=ctx)
    assert kernel_dict.load() is not None


def test_atmosphere_homogeneous_width(mode_mono):
    # Automatic width computation works as intended
    r = HomogeneousAtmosphere(toa_altitude=10.0 * ureg.km)
    ctx = KernelDictContext()
    wavelength = ctx.spectral_ctx.wavelength
    assert np.isclose(
        r.kernel_width(ctx), 10.0 / compute_sigma_s_air(wavelength=wavelength)
    )


@pytest.mark.parametrize("ref", (False, True))
def test_atmosphere_homogeneous_kernel_dict(mode_mono, ref):
    from mitsuba.core.xml import load_dict

    r = HomogeneousAtmosphere()

    # Default kernel dict constructs can be loaded by the kernel
    ctx = KernelDictContext(ref=False)

    dict_phase = onedict_value(r.kernel_phase(ctx))
    assert load_dict(dict_phase) is not None

    dict_medium = onedict_value(r.kernel_media(ctx))
    assert load_dict(dict_medium) is not None

    dict_shape = onedict_value(r.kernel_shapes(ctx))
    assert load_dict(dict_shape) is not None

    # Produced scene can be instantiated
    ctx = KernelDictContext(ref=ref)
    kernel_dict = KernelDict.new(r, ctx=ctx)
    assert kernel_dict.load() is not None
