from abc import ABC, abstractmethod
from copy import deepcopy

import attr


@attr.s
class Surface(ABC):
    """An abstract base class defining common facilities for all surfaces."""

    @abstractmethod
    def bsdfs(self):
        """Return phase function plugin interfaces.

        return (list): List of ``Bsdf`` plugin interfaces.
        """
        pass

    @abstractmethod
    def shapes(self):
        """Return shape plugin interfaces using references.

        return (list): List of ``Shape`` plugin interfaces.
        """
        pass

    def add_to(self, dict_scene, inplace=False):
        """Add current surface to scene dictionary.
        """
        if not inplace:  # If copy mode, replace mutable argument with a copy of itself
            dict_scene = deepcopy(dict_scene)

        dict_scene["surface"] = self.shapes()["shape_surface"]

        return dict_scene


@attr.s
class Lambertian(Surface):
    r"""This class builds a Lambertian surface.

    Constructor arguments / public attributes:
        ``reflectance`` (float):
            Surface reflectance value [dimensionless].
        ``width`` (float):
            Surface width [m].
    """

    reflectance = attr.ib(default=0.5)
    width = attr.ib(default=1.)

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        """(Re)initialise hidden internal state
        """
        self._bsdf = None
        self._shape = None

    def bsdfs(self):
        if self._bsdf is None:
            self._bsdf = {
                "bsdf_surface": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "uniform",
                        "value": self.reflectance
                    }
                }
            }

        return self._bsdf

    def shapes(self):  # TODO: add support of referencing
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if self._shape is None:
            bsdf = self.bsdfs()["bsdf_surface"]

            self._shape = {
                "shape_surface": {
                    "type": "rectangle",
                    "to_world": ScalarTransform4f
                        .scale(ScalarVector3f(self.width / 2., self.width / 2., 1.)),
                    "bsdf": bsdf
                }
            }

        return self._shape
