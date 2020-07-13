from abc import ABC, abstractmethod

import attr
import cerberus

from .collections import configdict


@attr.s
class ConfigObject(ABC):
    """Abstract base class for objects configured using a validation schema."""

    @property
    @abstractmethod
    def CONFIG_SCHEMA(self):
        """TODO: add docs"""
        pass

    config = attr.ib(default={})

    def __attrs_post_init__(self):
        # Check config contents and normalise
        v = cerberus.Validator(self.CONFIG_SCHEMA)

        if not v.validate(self.config):
            raise ValueError(v.errors)

        self.config = configdict(v.normalized(self.config))
