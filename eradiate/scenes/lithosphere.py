"""Lithosphere-related scene generation facilities.

.. admonition:: Registered factory members
    :class: hint

    .. factorytable::
       :factory: SceneElementFactory
       :modules: eradiate.scenes.lithosphere
"""

from abc import ABC, abstractmethod

import attr

from .core import SceneElement, SceneElementFactory
from .illumination import _validator_has_quantity
from .spectra import Spectrum, UniformReflectanceSpectrum
from ..util.attrs import attrib_quantity, validator_is_positive
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


@attr.s
class Surface(SceneElement, ABC):
    """An abstract base class defining common facilities for all surfaces.
    All these surfaces consist of a square parametrised by its width.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``width`` (float):
        Surface size. Default: 100 km.

        Unit-enabled field (default: cdu[length]).
    """

    id = attr.ib(
        default="surface",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    width = attrib_quantity(
        default=ureg.Quantity(100., ureg.km),
        validator=validator_is_positive,
        units_compatible=cdu.generator("length")
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

        width = self.width.to(kdu.get("length")).magnitude

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


@SceneElementFactory.register(name="lambertian")
@attr.s
class LambertianSurface(Surface):
    """Lambertian surface scene element [:factorykey:`lambertian`].

    This class creates a square surface to which a Lambertian BRDF is attached.

    See :class:`.Surface` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``reflectance`` (:class:`.UniformSpectrum`):
        Reflectance spectrum.
        Default: ``UniformReflectanceSpectrum(value=0.5)``.

        Can be initialised with a dictionary processed by
        :class:`.SceneElementFactory`.
    """

    reflectance = attr.ib(
        default=attr.Factory(lambda: UniformReflectanceSpectrum(value=0.5)),
        converter=Spectrum.converter("reflectance"),
        validator=[attr.validators.instance_of(Spectrum),
                   _validator_has_quantity("reflectance")]
    )

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": self.reflectance.kernel_dict()["spectrum"]
            }
        }


@SceneElementFactory.register(name="black")
@attr.s
class BlackSurface(Surface):
    """Black surface scene element [:factorykey:`black`].

    This class creates a square surface with a non reflecting BRDF attached.

    See :class:`.Surface` for undocumented members.
    """

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0}
            }
        }


@SceneElementFactory.register(name="rpv")
@attr.s
class RPVSurface(Surface):
    """RPV surface scene element [:factorykey:`rpv`].

    This class creates a square surface to which a RPV BRDF
    :cite:`Rahman1993CoupledSurfaceatmosphereReflectance`
    is attached.

    The default configuration corresponds to grassland (visible light)
    (:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`, Table 1).

    See :class:`.Surface` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``rho_0`` (float):
        Default: 0.183.

    ``k`` (float):
        Default: 0.780.

    ``ttheta`` (float):
        Default: -0.1.
    """

    # TODO: check if there are bounds to default parameters
    # TODO: add support for spectra

    rho_0 = attr.ib(
        default=0.183,
        converter=float
    )

    k = attr.ib(
        default=0.780,
        converter=float
    )

    ttheta = attr.ib(
        default=-0.1,
        converter=float
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
                "ttheta": {
                    "type": "uniform",
                    "value": self.ttheta
                }
            }
        }
