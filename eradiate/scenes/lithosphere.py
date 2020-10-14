"""Lithosphere-related scene generation facilities.

.. admonition:: Factory-enabled scene elements
    :class: hint

    .. factorytable::
        :modules: lithosphere
"""

from abc import ABC, abstractmethod

import attr

from .core import SceneElementFactory, SceneElement
from .spectra import UniformSpectrum
from ..util.attrs import attrib, attrib_float_positive, attrib_units
from ..util.units import config_default_units as cdu, ureg
from ..util.units import kernel_default_units as kdu


@attr.s
class Surface(SceneElement, ABC):
    """An abstract base class defining common facilities for all surfaces.
    """

    id = attrib(
        default="surface",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    width = attrib_float_positive(
        default=1.,
        has_units=True
    )
    width_units = attrib_units(
        compatible_units=ureg.m,
        default=attr.Factory(lambda: cdu.get("length"))
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

        width = self.get_quantity("width").to(kdu.get("length")).magnitude

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

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: python

               {
                   "width": 1.,
                   "reflectance": {
                       "type": "uniform",
                       "value": 0.5
                   }
               }

    .. admonition:: Configuration format
        :class: hint

        ``width`` (float):
            Size of the square surface [u_length].

            Default: 1.

        ``reflectance`` (dict):
            Reflectance spectrum [dimensionless].
            This section must be a factory configuration dictionary which will
            be passed to :meth:`.Factory.create`.

            Allowed scene elements:
            :factorykey:`uniform` (if selected, ``value`` must be in [0, 1]).

            Default:
            :factorykey:`uniform` with ``value`` set to 0.5.
    """

    reflectance = attrib(
        default=attr.Factory(lambda: UniformSpectrum(quantity="reflectance", value=0.5)),
        converter=SceneElementFactory.convert,
        validator=attr.validators.instance_of(UniformSpectrum),
    )

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "diffuse",
                "reflectance": self.reflectance.kernel_dict()["spectrum"]
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

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "width": 1.,
                   "rho_0": 0.183,
                   "k": 0.78,
                   "ttheta": -0.1,
               }

    .. admonition:: Configuration format
        :class: hint

        ``rho_0`` (float):
            Default: 0.183.

        ``k`` (float):
            Default: 0.780.

        ``ttheta`` (float):
            Default: -0.1.

        ``width`` (float):
            Size of the square surface [u_length].

            Default: 1.
    """

    # TODO: check if there are bounds to default parameters
    # TODO: add support for spectra

    rho_0 = attrib(
        default=0.183,
        converter=float
    )

    k = attrib(
        default=0.780,
        converter=float
    )

    ttheta = attrib(
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
