import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.kernel import mi_render, mi_traverse
from eradiate.scenes.biosphere import LeafCloud
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.integrators import PathIntegrator
from eradiate.scenes.measure import PerspectiveCameraMeasure
from eradiate.scenes.surface import BasicSurface

# TODO: Unit tests
# TODO: Benchmark


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
    mi_scene = mi_traverse(mi.load_dict(template.render(ctx=KernelDictContext())))
    contexts = [KernelDictContext(spectral_ctx={"wavelength": w}) for w in wavelengths]
    result = mi_render(mi_scene, contexts, spp=spp)

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
    kdict = template.render(ctx=KernelDictContext(), drop=True)

    for w in wavelengths:
        mi_scene = mi_traverse(mi.load_dict(kdict))
        mi_render(
            mi_scene,
            [KernelDictContext(spectral_ctx={"wavelength": w})],
            spp=spp,
        )
