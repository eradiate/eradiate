import numpy as np
import pint
import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.radprops.rayleigh import compute_sigma_s_air
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.core import KernelDict
from eradiate.scenes.phase import RayleighPhaseFunction, phase_function_factory


def test_homogeneous_default(mode_mono):
    """
    Applies default attributes values.
    """
    r = HomogeneousAtmosphere()
    assert r.bottom == 0.0 * ureg.km
    assert r.top == 10.0 * ureg.km
    assert isinstance(r.phase, RayleighPhaseFunction)


def test_homogeneous_sigma_s(mode_mono):
    """
    Assigns custom 'sigma_s' value.
    """
    spectral_ctx = SpectralContext.new()
    r = HomogeneousAtmosphere(sigma_s=1e-5)
    assert r.eval_sigma_s(spectral_ctx) == ureg.Quantity(1e-5, ureg.m ** -1)


def test_homogeneous_top(mode_mono):
    """
    Assigns custom 'top' value.
    """
    r = HomogeneousAtmosphere(top=8.0 * ureg.km)
    assert r.top == 8.0 * ureg.km


def test_homogeneous_top_invalid_units(mode_mono):
    """
    Raises when invalid units are passed to 'top'.
    """
    with pytest.raises(pint.errors.DimensionalityError):
        HomogeneousAtmosphere(top=10 * ureg.s)


def test_homogeneous_width_invalid_units(mode_mono):
    """
    Raises when invalid units are passed to 'width'.
    """
    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(width=5 * ureg.m ** 2)


def test_homogeneous_sigma_s_invalid_units(mode_mono):
    """
    Raises when invalid units are passed to 'sigma_s'.
    """
    with pytest.raises(pinttr.exceptions.UnitsError):
        HomogeneousAtmosphere(sigma_s=1e-7 * ureg.m)


def test_homogeneous_top_invalid_value(mode_mono):
    """
    Raises when invalid value is passed to 'top'.
    """
    with pytest.raises(ValueError):
        HomogeneousAtmosphere(top=-100.0)


def test_homogeneous_top_invalid_value(mode_mono):
    """
    Raises when invalid value is passed to 'width'.
    """
    with pytest.raises(ValueError):
        HomogeneousAtmosphere(width=-50.0)


@pytest.mark.parametrize(
    "phase_id",
    set(phase_function_factory.registry.keys())
    - {
        "blend_phase",
        "tab_phase",
    },  # Exclude phase functions with no default parametrisation
)
@pytest.mark.parametrize("ref", (False, True))
def test_homogeneous_phase_function(mode_mono, phase_id, ref):
    """Supports all available phase function types."""
    r = HomogeneousAtmosphere(phase={"type": phase_id})

    # The resulting object produces a valid kernel dictionary
    ctx = KernelDictContext(ref=ref)
    kernel_dict = KernelDict.from_elements(r, ctx=ctx)
    assert kernel_dict.load() is not None


def test_homogeneous_width(mode_mono):
    """
    Automatically sets width to ten times the scattering mean free path.
    """
    r = HomogeneousAtmosphere()
    ctx = KernelDictContext()
    wavelength = ctx.spectral_ctx.wavelength
    assert np.isclose(
        r.kernel_width(ctx), 10.0 / compute_sigma_s_air(wavelength=wavelength)
    )


@pytest.mark.parametrize("ref", (False, True))
def test_homogeneous_kernel_dict(mode_mono, ref):
    """
    Produces kernel dictionaries that can be loaded by the kernel.
    """
    from mitsuba.core.xml import load_dict

    r = HomogeneousAtmosphere()

    ctx = KernelDictContext(ref=False)

    dict_phase = onedict_value(r.kernel_phase(ctx))
    assert load_dict(dict_phase) is not None

    dict_medium = onedict_value(r.kernel_media(ctx))
    assert load_dict(dict_medium) is not None

    dict_shape = onedict_value(r.kernel_shapes(ctx))
    assert load_dict(dict_shape) is not None

    ctx = KernelDictContext(ref=ref)
    kernel_dict = KernelDict.from_elements(r, ctx=ctx)
    assert kernel_dict.load() is not None
