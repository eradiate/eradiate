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
