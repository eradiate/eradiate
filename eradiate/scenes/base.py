"""Basic abstractions and utilities to assist with scene generation."""

from abc import ABC, abstractmethod
from copy import deepcopy

import attr


@attr.s
class SceneHelper(ABC):
    """TODO: add docs"""
    
    @property
    @abstractmethod
    def DEFAULT_CONFIG(self):
        """TODO: add docs"""
        pass

    config = attr.ib(default={})
    id = attr.ib(default=None)

    def __attrs_post_init__(self):
        # Merge configuration with defaults
        self.config = {**self.DEFAULT_CONFIG, **self.config}
        # Initialise internal state
        self.init()

    def init(self):
        """(Re)initialise internal state."""
        pass

    @abstractmethod
    def kernel_dict(self, ref=True):
        """Return dictionary suitable for kernel scene configuration.

        Parameter ``ref`` (bool):
            If `True`, use referencing for all relevant nested plugins.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        pass

    def add_to(self, scene_dict, inplace=False, ref=True):
        """Merge two kernel scene dictionaries.

        This method calls :meth:`kernel_dict` and merges the resulting
        dictionary with ``scene_dict``.

        Parameter ``scene_dict`` (dict):
            Dictionary suitable to instantiate a :class:`mitsuba.render.Scene`
            (using :func:`~mitsuba.core.xml.load_dict`).

        Parameter ``inplace`` (bool):
            If `True`, the passed scene dictionary will be modified in-place.
            Otherwise, the returned dictionary will be a copy of ``scene_dict``.

        Parameter ``ref`` (bool):
            If `True`, plugins eligible to referencing will be referenced.

        Returns → dict:
            Modified scene dictionary.
        """

        if not inplace:  # If copy mode, replace mutable argument with a copy of itself
            scene_dict = deepcopy(scene_dict)

        for key, value in self.kernel_dict(ref=ref).items():
            scene_dict[key] = value

        return scene_dict

    @classmethod
    def from_dict(cls, d):
        """Create from a configuration dictionary.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation. The configuration
            dictionary uses the same structure as :data:`DEFAULT_CONFIG`.

        Returns → :class:`~eradiate.scenes.base.SceneHelper`:
            Created object.
        """
        return cls(config=d)


def scene_dict_empty():
    """Return an empty Mitsuba scene dictionary."""
    return {"type": "scene"}
