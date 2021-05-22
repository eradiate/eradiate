import attr

from ._core import Surface, SurfaceFactory
from ..._attrs import documented, parse_docs
from ...contexts import KernelDictContext


@SurfaceFactory.register("rpv")
@parse_docs
@attr.s
class RPVSurface(Surface):
    """
    RPV surface scene element [:factorykey:`rpv`].

    This class creates a square surface to which a RPV BRDF
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance`
    is attached.

    The default configuration corresponds to grassland (visible light)
    (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).
    """

    # TODO: check if there are bounds to default parameters
    # TODO: match defaults with plugin defaults
    # TODO: add support for spectra

    rho_0 = documented(
        attr.ib(default=0.183, converter=float),
        doc=":math:`\\rho_0` parameter.",
        type="float",
        default="0.183",
    )

    k = documented(
        attr.ib(default=0.780, converter=float),
        doc=":math:`k` parameter.",
        type="float",
        default="0.780",
    )

    ttheta = documented(
        attr.ib(default=-0.1, converter=float),
        doc=":math:`\\Theta` parameter.",
        type="float",
        default="-0.1",
    )

    def bsdfs(self, ctx: KernelDictContext = None):
        return {
            f"bsdf_{self.id}": {
                "type": "rpv",
                "rho_0": {"type": "uniform", "value": self.rho_0},
                "k": {"type": "uniform", "value": self.k},
                "g": {"type": "uniform", "value": self.ttheta},
            }
        }
