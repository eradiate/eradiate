import attr

from ._core import BSDF, bsdf_factory
from ..core import KernelDict
from ...contexts import KernelDictContext


@bsdf_factory.register(type_id="black")
@attr.s
class BlackBSDF(BSDF):
    """
    Black BSDF [``black``].

    This BSDF models a perfectly absorbing surface. It is equivalent to a
    :class:`.DiffuseBSDF` with zero reflectance.

    Notes
    -----
    This is a thin wrapper around the ``diffuse`` kernel plugin.
    """

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Inherit docstring
        return KernelDict(
            {
                self.id: {
                    "type": "diffuse",
                    "reflectance": {"type": "uniform", "value": 0.0},
                }
            }
        )
