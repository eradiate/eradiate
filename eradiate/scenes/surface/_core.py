from abc import ABC, abstractmethod
from copy import deepcopy

import attr
import pinttr

from ..core import SceneElement
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory
from ..._units import unit_context_config as ucc
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg


@parse_docs
@attr.s
class Surface(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all surfaces.
    All these surfaces consist of a square parametrised by its width.
    """

    id = documented(
        attr.ib(
            default="surface",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"surface"',
    )

    width = documented(
        pinttr.ib(
            default=ureg.Quantity(100.0, ureg.km),
            validator=validators.is_positive,
            units=ucc.deferred("length"),
        ),
        doc="Surface size.\n\nUnit-enabled field (default: cdu[length]).",
        type="float",
        default="100 km",
    )

    @abstractmethod
    def bsdfs(self):
        """Return BSDF plugin specifications only.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the surface.
        """
        # TODO: return a KernelDict
        pass

    def shapes(self, ref=False):
        """Return shape plugin specifications only.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            attached to the surface.
        """
        from mitsuba.core import ScalarTransform4f, ScalarVector3f

        if ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs()[f"bsdf_{self.id}"]

        width = self.width.to(uck.get("length")).magnitude

        return {
            f"shape_{self.id}": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(
                    ScalarVector3f(width * 0.5, width * 0.5, 1.0)
                ),
                "bsdf": bsdf,
            }
        }

    def kernel_dict(self, ref=True):
        kernel_dict = {}

        if not ref:
            kernel_dict[self.id] = self.shapes(ref=False)[f"shape_{self.id}"]
        else:
            kernel_dict[f"bsdf_{self.id}"] = self.bsdfs()[f"bsdf_{self.id}"]
            kernel_dict[self.id] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict

    def scaled(self, factor):
        """
        Return a copy of self scaled by a given factor.
        """
        return attr.evolve(self, width=self.width * factor)


class SurfaceFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Surface`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: SurfaceFactory
    """

    _constructed_type = Surface
    registry = {}
