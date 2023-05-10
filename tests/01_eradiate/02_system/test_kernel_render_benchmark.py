import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.kernel import mi_render, mi_traverse
from eradiate.scenes.biosphere import LeafCloud
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.integrators import PathIntegrator
from eradiate.scenes.measure import PerspectiveCameraMeasure
from eradiate.scenes.surface import BasicSurface

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
    Mitsuba scene render
    ====================

    This basic test checks that the ``mi_render()`` function runs without
    failing and stores results as expected.

    Rationale
    ---------

    We construct a simple scene and render it using our ``mi_render()`` wrapper
    for a sequence of wavelengths.

    Expected behaviour
    ------------------

    Rendering succeed and results are stored in a dictionary which uses
    parametric loop index values (in this case, the wavelength) as keys.
    """
    template, params = traverse(make_scene())
    mi_scene = mi_traverse(mi.load_dict(template.render(ctx=KernelContext())))
    contexts = [KernelContext(si={"w": w}) for w in wavelengths]
    result = mi_render(mi_scene, contexts, spp=spp)

    # We store film values and SPPs
    assert set(result.keys()) == set(x.si.w.magnitude for x in contexts)
    assert set(result[500.0].keys()) == {"measure"}


@pytest.mark.slow
def test_mi_render_rebuild(mode_mono):
    """
    Mitsuba render scene rebuild
    ============================

    This test performs the same computation as the previous (see
    *Mitsuba scene render*), but rebuilds the scene for each parametric
    iteration. It is designed for comparison with the previous.

    Rationale
    ---------

    The *Mitsuba scene render* test does not reload the scene at each iteration;
    instead, it uses Mitsuba's parameter update system to update it at a low
    cost. In this test, the scene is entirely rebuilt at each iteration.

    Expected behaviour
    ------------------

    The rendering sequence succeeds and yields the same results as the
    *Mitsuba scene render* test. This test should take much longer than the
    *Mitsuba render scene* to complete.
    """
    # TODO: Recycle this test and merge with previous
    template, params = traverse(make_scene())
    kdict = template.render(ctx=KernelContext(), drop=True)

    for w in wavelengths:
        mi_scene = mi_traverse(mi.load_dict(kdict))
        mi_render(
            mi_scene,
            [KernelContext(si={"w": w})],
            spp=spp,
        )
