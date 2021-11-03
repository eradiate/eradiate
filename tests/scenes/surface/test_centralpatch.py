import numpy as np
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.surface import CentralPatchSurface, LambertianSurface
from eradiate.units import unit_registry as ureg


def test_centralpatch_instantiate(mode_mono):
    ctx = KernelDictContext()

    # Default constructor
    cs = CentralPatchSurface()

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.from_elements(cs, ctx=ctx)
    assert kernel_dict.load() is not None

    # background width must be AUTO
    with pytest.raises(ValueError):
        cs = CentralPatchSurface(
            width=1000.0,
            central_patch=LambertianSurface(
                reflectance={"type": "uniform", "value": 0.3}
            ),
            background_surface=LambertianSurface(
                reflectance={"type": "uniform", "value": 0.3}, width=10 * ureg.m
            ),
        )

    # Constructor with arguments
    cs = CentralPatchSurface(
        width=1000.0,
        central_patch=LambertianSurface(reflectance={"type": "uniform", "value": 0.3}),
        background_surface=LambertianSurface(
            reflectance={"type": "uniform", "value": 0.8}
        ),
    )

    # Check if produced scene can be instantiated
    assert KernelDict.from_elements(cs, ctx=ctx).load() is not None


def test_centralpatch_compute_scale(modes_all_double):
    ctx = KernelDictContext()
    cs = CentralPatchSurface(
        width=10 * ureg.m, central_patch=LambertianSurface(width=1 * ureg.m)
    )

    scale = cs._compute_scale_parameter(ctx=ctx)

    assert np.allclose(scale, 3.33333333)

    ctx = ctx.evolve(override_canopy_width=2 * ureg.m)
    scale = cs._compute_scale_parameter(ctx=ctx)
    assert np.allclose(scale, 1.6666666666667)


def test_centralpatch_scale_kernel_dict(mode_mono):
    from mitsuba.core import ScalarTransform4f

    cs = CentralPatchSurface(
        width=3000.0 * ureg.km,
        central_patch=LambertianSurface(width=100 * ureg.km),
        id="surface",
    )

    ctx = KernelDictContext()

    kernel_dict = cs.bsdfs(ctx=ctx)

    assert np.allclose(
        kernel_dict["bsdf_surface"]["weight"]["to_uv"].matrix,
        (
            ScalarTransform4f.scale(10) * ScalarTransform4f.translate((-0.45, -0.45, 0))
        ).matrix,
    )
