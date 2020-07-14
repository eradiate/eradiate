from abc import abstractmethod

import attr

from ..core import SceneHelper


@attr.s
class Atmosphere(SceneHelper):
    """An abstract base class defining common facilities for all atmospheres."""

    id = attr.ib(default="atmosphere")

    @abstractmethod
    def phase(self):
        """Return phase function plugin specifications only.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the phase
            functions attached to the atmosphere.
        """
        # TODO: return a KernelDict
        pass

    @abstractmethod
    def media(self, ref=False):
        """Return medium plugin specifications only.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the media
            attached to the atmosphere.
        """
        # TODO: return a KernelDict
        pass

    @abstractmethod
    def shapes(self, ref=False):
        """Return shape plugin specifications only.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            attached to the atmosphere.
        """
        # TODO: return a KernelDict
        pass
    
    def kernel_dict(self, ref=True):
        # TODO: return a KernelDict
        kernel_dict = {"integrator": {"type": "volpath"}}  # Force volpath integrator
        
        if not ref:
            kernel_dict["atmosphere"] = self.shapes()["shape_atmosphere"]
        else:
            kernel_dict["phase_atmosphere"] = self.phase()["phase_atmosphere"]
            kernel_dict["medium_atmosphere"] = self.media(ref=True)["medium_atmosphere"]
            kernel_dict["atmosphere"] = self.shapes(ref=True)["shape_atmosphere"]

        return kernel_dict
