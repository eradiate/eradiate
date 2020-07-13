from abc import abstractmethod

import attr

from .. import SceneHelper


@attr.s
class Atmosphere(SceneHelper):
    """An abstract base class defining common facilities for all atmospheres."""

    id = attr.ib(default="atmosphere")

    @abstractmethod
    def phase(self):
        # TODO: add docs
        pass

    @abstractmethod
    def media(self, ref=False):
        # TODO: add docs
        pass

    @abstractmethod
    def shapes(self, ref=False):
        # TODO: add docs
        pass
    
    def kernel_dict(self, ref=True):
        kernel_dict = {"integrator": {"type": "volpath"}}  # Force volpath integrator
        
        if not ref:
            kernel_dict["atmosphere"] = self.shapes()["shape_atmosphere"]
        else:
            kernel_dict["phase_atmosphere"] = self.phase()["phase_atmosphere"]
            kernel_dict["medium_atmosphere"] = self.media(ref=True)["medium_atmosphere"]
            kernel_dict["atmosphere"] = self.shapes(ref=True)["shape_atmosphere"]

        return kernel_dict
