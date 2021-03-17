"""Surface-related scene generation facilities.

.. admonition:: Registered factory members [:class:`SurfaceFactory`]
   :class: hint

   .. factorytable::
      :factory: SurfaceFactory
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import attr
import pinttr

from .core import SceneElement
from .spectra import Spectrum, SpectrumFactory
from .._attrs import documented, get_doc, parse_docs
from .._factory import BaseFactory
from .._units import unit_context_config as ucc
from .._units import unit_context_kernel as uck
from .._units import unit_registry as ureg
from ..validators import has_quantity, is_positive


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
        default="\"surface\""
    )

    width = documented(
        pinttr.ib(
            default=ureg.Quantity(100., ureg.km),
            validator=is_positive,
            units=ucc.deferred("length")
        ),
        doc="Surface size.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
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
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs()[f"bsdf_{self.id}"]

        width = self.width.to(uck.get("length")).magnitude

        return {
            f"shape_{self.id}": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(ScalarVector3f(
                    width * 0.5, width * 0.5, 1.)
                ),
                "bsdf": bsdf
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
        """Return a copy of self scaled by a given factor."""
        result = deepcopy(self)
        result.width *= factor
        return result


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


@SurfaceFactory.register("lambertian")
@parse_docs
@attr.s
class LambertianSurface(Surface):
    """Lambertian surface scene element [:factorykey:`lambertian`].

    This class creates a square surface to which a Lambertian BRDF is attached.
    """

    reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[attr.validators.instance_of(Spectrum),
                       has_quantity("reflectance")]
        ),
        doc="Reflectance spectrum. Can be initialised with a dictionary "
            "processed by :class:`.SpectrumFactory`.",
        type=":class:`.UniformSpectrum`",
        default="0.5",
    )

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": self.reflectance.kernel_dict()["spectrum"]
            }
        }


@SurfaceFactory.register("black")
@parse_docs
@attr.s
class BlackSurface(Surface):
    """Black surface scene element [:factorykey:`black`].

    This class creates a square surface with a black BRDF attached.
    """

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.}
            }
        }


@SurfaceFactory.register("rpv")
@parse_docs
@attr.s
class RPVSurface(Surface):
    """RPV surface scene element [:factorykey:`rpv`].

    This class creates a square surface to which a RPV BRDF
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance`
    is attached.

    The default configuration corresponds to grassland (visible light)
    (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).
    """

    # TODO: check if there are bounds to default parameters
    # TODO: add support for spectra

    rho_0 = documented(
        attr.ib(
            default=0.183,
            converter=float
        ),
        doc=":math:`\\rho_0` parameter.",
        type="float",
        default="0.183",
    )

    k = documented(
        attr.ib(
            default=0.780,
            converter=float
        ),
        doc=":math:`k` parameter.",
        type="float",
        default="0.780",
    )

    ttheta = documented(
        attr.ib(
            default=-0.1,
            converter=float
        ),
        doc=":math:`\\Theta` parameter.",
        type="float",
        default="-0.1",
    )

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "rpv",
                "rho_0": {
                    "type": "uniform",
                    "value": self.rho_0
                },
                "k": {
                    "type": "uniform",
                    "value": self.k
                },
                "g": {
                    "type": "uniform",
                    "value": self.ttheta
                }
            }
        }
