import numpy as np
import pint
import pinttr
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.radprops.rayleigh import compute_sigma_s_air
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
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
    assert r.eval_sigma_s(spectral_ctx) == ureg.Quantity(1e-5, ureg.m**-1)


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


@pytest.mark.parametrize(
    "phase_id",
    set(phase_function_factory.registry.keys())
    - {
        "blend_phase",
        "tab_phase",
    },  # Exclude phase functions with no default parametrisation
)
def test_homogeneous_phase_function(mode_mono, phase_id):
    """Supports all available phase function types."""
    r = HomogeneousAtmosphere(geometry="plane_parallel", phase={"type": phase_id})

    # The resulting object produces a valid kernel dictionary
    ctx = KernelDictContext()
    assert r.kernel_dict(ctx).load()


def test_homogeneous_mfp(mode_mono):
    """
    Automatically sets width to ten times the scattering mean free path.
    """
    ctx = KernelDictContext()
    r = HomogeneousAtmosphere(geometry="plane_parallel")
    sigma_s = compute_sigma_s_air(wavelength=ctx.spectral_ctx.wavelength)

    assert np.isclose(r.eval_mfp(ctx), 1.0 / sigma_s)
    assert np.isclose(r.kernel_width_plane_parallel(ctx), 10.0 / sigma_s)


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_homogeneous_kernel_dict(modes_all_double, geometry):
    """
    Produces kernel dictionaries that can be loaded by the kernel.
    """

    r = HomogeneousAtmosphere(geometry=geometry)
    ctx = KernelDictContext()

    assert r.kernel_phase(ctx).load()
    assert r.kernel_media(ctx).load()
    assert r.kernel_shapes(ctx).load()
    assert r.kernel_dict(ctx).load()
