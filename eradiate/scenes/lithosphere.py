"""Lithosphere-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: lithosphere
"""

from abc import abstractmethod

import attr

from . import SceneHelper
from .core import Factory
from ..util.collections import frozendict


@attr.s
class Surface(SceneHelper):
    """An abstract base class defining common facilities for all surfaces."""

    id = attr.ib(default="surface")

    @abstractmethod
    def bsdfs(self):
        """TODO: add docs"""
        pass

    @abstractmethod
    def shapes(self, ref=False):
        """TODO: add docs"""
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

    .. admonition:: Configuration format
        :class: hint

        ``reflectance`` (float):
            Reflectance [dimensionless].

            Default value: 0.5.

        ``width`` (float):
            Size of the square surface [u_length].

            Default: 1.
    """

    CONFIG_SCHEMA = frozendict({
        "reflectance": {
            "type": "number",
            "min": 0.,
            "max": 1.,
            "default": 0.5,
        },
        "width": {
            "type": "number",
            "min": 0.,
            "default": 1.,
        }
    })

    def bsdfs(self):
        return {
            "bsdf_surface": {
                "type": "diffuse",
                "reflectance": {
                    "type": "uniform",
                    "value": self.config["reflectance"]
                }
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if ref:
            bsdf = {"type": "ref", "id": "bsdf_surface"}
        else:
            bsdf = self.bsdfs()["bsdf_surface"]

        size = self.config["width"]

        return {
            "shape_surface": {
                "type": "rectangle",
                "to_world": ScalarTransform4f
                    .scale(ScalarVector3f(size / 2., size / 2., 1.)),
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

    CONFIG_SCHEMA = frozendict({
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

        size = self.config["width"]

        return {
            "shape_surface": {
                "type": "rectangle",
                "to_world": ScalarTransform4f
                    .scale(ScalarVector3f(size / 2., size / 2., 1.)),
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
