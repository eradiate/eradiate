"""Lithosphere-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: lithosphere
"""

from abc import abstractmethod

import attr

from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg
from .core import Factory, SceneHelper


@attr.s
class Surface(SceneHelper):
    """An abstract base class defining common facilities for all surfaces.
    """

    id = attr.ib(default="surface")

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

    @abstractmethod
    def shapes(self, ref=False):
        """Return shape plugin specifications only.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            attached to the surface.
        """
        # TODO: return a KernelDict
        pass

    def kernel_dict(self, ref=True):
        kernel_dict = {}

        if not ref:
            kernel_dict["surface"] = self.shapes(ref=False)["shape_surface"]
        else:
            kernel_dict["bsdf_surface"] = self.bsdfs()["bsdf_surface"]
            kernel_dict["surface"] = self.shapes(ref=True)["shape_surface"]

        return kernel_dict


@attr.s
@Factory.register(name="lambertian")
class LambertianSurface(Surface):
    """Lambertian surface scene generation helper [:factorykey:`lambertian`].

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

            Allowed scene generation helpers:
            :factorykey:`uniform` (if selected, ``value`` must be in [0, 1]).

            Default:
            :factorykey:`uniform` with ``value`` set to 0.5.
    """

    @classmethod
    def config_schema(cls):
        return dict({
            "reflectance": {
                "type": "dict",
                "default": {},
                "allow_unknown": True,
                "schema": {
                    "type": {
                        "type": "string",
                        "allowed": ["uniform"],
                        "default": "uniform"
                    },
                    "value": {  # If selecting uniform, we check that this is a reflectance spectrum
                        "type": "number",
                        "dependencies": {"type": "uniform"},
                        "required": False,
                        "min": 0.,
                        "max": 1.,
                        "default": 0.5
                    },
                    "quantity": {
                        "type": "string",
                        "nullable": True,
                        "allowed": [],
                        "default": None
                    }
                },
            },
            "width": {
                "type": "number",
                "min": 0.,
                "default": 1.,
            },
            "width_unit": {
                "type": "string",
                "default": cdu.get_str("length")
            }
        })

    def bsdfs(self):
        reflectance = Factory().create(self.config["reflectance"])
        return {
            "bsdf_surface": {
                "type": "diffuse",
                "reflectance": reflectance.kernel_dict()["spectrum"]
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if ref:
            bsdf = {"type": "ref", "id": "bsdf_surface"}
        else:
            bsdf = self.bsdfs()["bsdf_surface"]

        width = self.config.get_quantity("width").to(kdu.get("length")).magnitude

        return {
            "shape_surface": {
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
            kernel_dict["surface"] = self.shapes(ref=False)["shape_surface"]
        else:
            kernel_dict["bsdf_surface"] = self.bsdfs()["bsdf_surface"]
            kernel_dict["surface"] = self.shapes(ref=True)["shape_surface"]

        return kernel_dict


@attr.s
@Factory.register(name="rpv")
class RPVSurface(Surface):
    """RPV surface scene generation helper [:factorykey:`rpv`].

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

    @classmethod
    def config_schema(cls):
        return dict({
            "rho_0": {
                "type": "number",
                "default": 0.183,
            },
            "k": {
                "type": "number",
                "default": 0.780,
            },
            "ttheta": {
                "type": "number",
                "default": -0.1,
            },
            "width": {
                "type": "number",
                "min": 0.,
                "default": 1.,
            },
            "width_unit": {
                "type": "string",
                "default": cdu.get_str("length")
            }
        })

    def bsdfs(self):
        return {
            "bsdf_surface": {
                "type": "rpv",
                "rho_0": {
                    "type": "uniform",
                    "value": self.config["rho_0"]
                },
                "k": {
                    "type": "uniform",
                    "value": self.config["k"]
                },
                "ttheta": {
                    "type": "uniform",
                    "value": self.config["ttheta"]
                }
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if ref:
            bsdf = {"type": "ref", "id": "bsdf_surface"}
        else:
            bsdf = self.bsdfs()["bsdf_surface"]

        width = self.config.get_quantity("width").to(kdu.get("length")).magnitude

        return {
            "shape_surface": {
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
            kernel_dict["surface"] = self.shapes(ref=False)["shape_surface"]
        else:
            kernel_dict["bsdf_surface"] = self.bsdfs()["bsdf_surface"]
            kernel_dict["surface"] = self.shapes(ref=True)["shape_surface"]

        return kernel_dict
