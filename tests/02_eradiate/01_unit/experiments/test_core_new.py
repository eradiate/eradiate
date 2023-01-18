import mitsuba as mi
import numpy as np

import eradiate.kernel.logging
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.experiments._core_new import mi_render
from eradiate.scenes.biosphere import LeafCloud
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.integrators import PathIntegrator
from eradiate.scenes.measure import PerspectiveCameraMeasure
from eradiate.scenes.surface import BasicSurface

eradiate.kernel.logging.install_logging()

wavelengths = np.linspace(500, 600, 21) * ureg.nm
spp = 4


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


def test_render_retained(mode_mono):
    template, params = traverse(make_scene())
    mi_scene = mi.load_dict(template.render(ctx=KernelDictContext(), drop=True))
    mi_render(
        mi_scene,
        params,
        ctxs=[KernelDictContext(spectral_ctx={"wavelength": w}) for w in wavelengths],
        spp=spp,
    )


def test_render_rebuild(mode_mono):
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
