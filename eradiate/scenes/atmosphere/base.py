from abc import abstractmethod

import attr

from ..base import SceneHelper


@attr.s
class Atmosphere(SceneHelper):
    """An abstract base class defining common facilities for all atmospheres."""

    id = attr.ib(default="atmosphere")

    @abstractmethod
    def phase(self):
        """TODO: add docs"""
        pass

    @abstractmethod
    def media(self, ref=False):
        """"TODO: add docs"""
        pass

    @abstractmethod
    def shapes(self, ref=False):
        """TODO: add docs"""
        pass
    
    def kernel_dict(self, ref=True):
        scene_dict = {"integrator": {"type": "volpath"}}  # Force volpath integrator
        
        if not ref:
            scene_dict["atmosphere"] = self.shapes()["shape_atmosphere"]
        else:
            scene_dict["phase_atmosphere"] = self.phase()["phase_atmosphere"]
            scene_dict["medium_atmosphere"] = self.media(ref=True)["medium_atmosphere"]
            scene_dict["atmosphere"] = self.shapes(ref=True)["shape_atmosphere"]

        return scene_dict
