"""Scene generation facilities related with the lithosphere."""

from abc import abstractmethod

import attr

from .base import SceneHelper
from .factory import Factory


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
        scene_dict = {}

        if not ref:
            scene_dict["surface"] = self.shapes(ref=False)["shape_surface"]
        else:
            scene_dict["bsdf_surface"] = self.bsdfs()["bsdf_surface"]
            scene_dict["surface"] = self.shapes(ref=True)["shape_surface"]

        return scene_dict


@attr.s
@Factory.register()
class Lambertian(Surface):
    r"""This class builds a Lambertian surface.
    """

    DEFAULT_CONFIG = {
        "reflectance": 0.5,
        "width": 1.,
    }

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
        scene_dict = {}

        if not ref:
            scene_dict["surface"] = self.shapes(ref=False)["shape_surface"]
        else:
            scene_dict["bsdf_surface"] = self.bsdfs()["bsdf_surface"]
            scene_dict["surface"] = self.shapes(ref=True)["shape_surface"]

        return scene_dict
