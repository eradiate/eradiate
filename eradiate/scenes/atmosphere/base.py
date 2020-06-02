from abc import ABC, abstractmethod
from copy import deepcopy

import attr


@attr.s
class Atmosphere(ABC):
    """An abstract base class defining common facilities for all atmospheres."""

    @abstractmethod
    def phase(self):
        """Return phase function plugin interfaces.

        Returns → list(dict):
            List of dictionaries suitable to instantiate
            :class:`mitsuba.render.Phase` plugins.
        """
        pass

    @abstractmethod
    def media(self, ref=False):
        """Return participating media plugin interfaces.

        Parameter ``ref`` (bool):
            If `True`, return nested plugins as references.

        Returns → list(dict):
            List of dictionaries suitable to instantiate
            :class:`mitsuba.render.Medium` plugins.
        """
        pass

    @abstractmethod
    def shapes(self, ref=False):
        """Return shape plugin interfaces using references.

        Parameter ``ref`` (bool):
            If `True`, return nested plugins as references.

        Returns → list(dict):
            List of dictionaries suitable to instantiate
            :class:`mitsuba.render.Shape` plugins.
        """
        pass

    def add_to(self, dict_scene, inplace=False, ref=True):
        """Add current atmosphere to scene dictionary.

        Parameter ``dict_scene`` (dict):
            Dictionary suitable to instantiate a :class:`mitsuba.render.Scene`.

        Parameter ``inplace`` (bool):
            If `True`, the passed scene dictionary will be modified in-place.
            Otherwise, the returned dictionary will be a copy of ``dict_scene``.

        Parameter ``ref`` (bool):
            If `True`, plugins eligible to referencing will be referenced.

        Returns → dict:
            Modified scene dictionary.
        """

        if not inplace:  # If copy mode, replace mutable argument with a copy of itself
            dict_scene = deepcopy(dict_scene)

        dict_scene["integrator"] = {"type": "volpath"}  # Force volpath integrator

        if not ref:
            dict_scene["atmosphere"] = self.shapes()["shape_atmosphere"]
        else:
            dict_scene["phase_atmosphere"] = self.phase()["phase_atmosphere"]
            dict_scene["medium_atmosphere"] = self.media(ref=True)["medium_atmosphere"]
            dict_scene["atmosphere"] = self.shapes(ref=True)["shape_atmosphere"]

        return dict_scene