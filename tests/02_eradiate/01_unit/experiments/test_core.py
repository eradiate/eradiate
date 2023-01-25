import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

import eradiate.kernel.logging
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.experiments import AtmosphereExperiment, mi_render
from eradiate.scenes.biosphere import LeafCloud
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.integrators import PathIntegrator
from eradiate.scenes.measure import PerspectiveCameraMeasure
from eradiate.scenes.surface import BasicSurface

eradiate.kernel.logging.install_logging()

wavelengths = np.linspace(500, 600, 21) * ureg.nm
spp = 100


def make_scene():
    return Scene(
        objects={
            "surface": BasicSurface(shape={"type": "rectangle"}),
            "canopy": LeafCloud.cuboid(
                n_leaves=10000,
                leaf_radius=5 * ureg.cm,
                l_horizontal=10 * ureg.m,
                l_vertical=1 * ureg.m,
            ),
            "illumination": DirectionalIllumination(),
            "integrator": PathIntegrator(),
            "measure": PerspectiveCameraMeasure(
                target=[0, 0, 0], origin=[2, 2, 2], up=[0, 0, 1]
            ),
        }
    )


def test_mi_render(mode_mono):
    """
    This basic test checks that the mi_render() function runs without failing
    and stores results as expected.
    """
    template, params = traverse(make_scene())
    mi_scene = mi.load_dict(template.render(ctx=KernelDictContext(), drop=True))
    contexts = [KernelDictContext(spectral_ctx={"wavelength": w}) for w in wavelengths]
    result = mi_render(
        mi_scene,
        params,
        ctxs=contexts,
        spp=spp,
    )

    # We store film values and SPPs
    assert set(result.keys()) == set(
        x.spectral_ctx.wavelength.magnitude for x in contexts
    )
    assert set(result[500.0].keys()) == {"measure"}


@pytest.mark.slow
def test_mi_render_rebuild(mode_mono):
    """
    This test performs the same computation as the previous, but rebuilds the
    scene for each parametric iteration. It is designed for comparison with the
    previous.
    """
    template, params = traverse(make_scene())
    kernel_dict = template.render(ctx=KernelDictContext(), drop=True)

    for w in wavelengths:
        mi_scene = mi.load_dict(kernel_dict)
        mi_render(
            mi_scene,
            params,
            ctxs=[KernelDictContext(spectral_ctx={"wavelength": w})],
            spp=spp,
        )


def test_run_function(modes_all_double):
    mode = eradiate.mode()

    measure = {"type": "mdistant"}
    if mode.is_mono:
        measure["spectral_cfg"] = {"wavelengths": [540.0, 550.0]}

    elif mode.is_ckd:
        measure["spectral_cfg"] = {"bin_set": "10nm", "bins": ["540", "550"]}

    else:
        assert False, f"Please add a test for mode '{mode.id}'"

    exp = AtmosphereExperiment(atmosphere=None, measures=measure)
    result = eradiate.run(exp)
    assert isinstance(result, xr.Dataset)
