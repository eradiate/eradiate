"""Configuration dictionary-based class components."""

from abc import ABC, abstractmethod

import attr
import cerberus

from .collections import ndict


@attr.s
class ConfigObject(ABC):
    """Abstract base class for objects configured using a validation schema.

    Every object inheriting from this class has a :data:`config` attribute which
    is intended to store a configuration dictionary.

    Upon execution of the constructor, the dictionary passed as an argument
    is validated using the `Cerberus <https://docs.python-cerberus.org>`_
    validation library. :data:`config` is checked for compliance with a
    validation schema returned by the :meth:`config_schema` class method.
    Should validation fail, a :class:`ValueError` is raised which contains the
    validation error report.

    If validation is successful, the constructor's :data:`config` argument is
    normalised to apply defaults defined in :meth:`config_schema`, then
    converted to a :class:`~eradiate.util.collections.configdict`.
    """

    @classmethod
    @abstractmethod
    def config_schema(cls):
        """Cerberus validation schema to validate :data:`config`.

        See the `Cerberus documentation <https://docs.python-cerberus.org>`_
        for available rules.

        .. note::

           This member is implemented as an abstract method to force defining
           it upon subclassing :class:`ConfigObject`.
        """
        pass

    config = attr.ib(default={})

    def __attrs_post_init__(self):
        # Check config contents and normalise
        v = cerberus.Validator(self.config_schema())

        if not v.validate(self.config):
            raise ValueError(v.errors)

        self.config = ndict(v.normalized(self.config))
