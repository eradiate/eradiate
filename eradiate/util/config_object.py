"""Configuration dictionary-based class components."""


from abc import ABC, abstractmethod

import attr
import cerberus
import pint

from .collections import configdict
from .units import ureg, DefaultUnits


@attr.s
class ConfigObject(ABC):
    """Abstract base class for objects configured using a validation schema.

    Every object inheriting from this class has a :data:`config` attribute which
    is intended to store a configuration dictionary.

    Upon execution of the constructor, the dictionary passed as an argument
    is validated using the `Cerberus <https://docs.python-cerberus.org>`_
    validation library. :data:`config` is checked for compliance with a
    validation schema stored in the :data:`CONFIG_SCHEMA` class constant.
    Should validation fail, a :class:`ValueError` is raised which contains the
    validation error report.

    If validation is successful, the constructor's :data:`config` argument is
    normalised to apply defaults defined in :data:`CONFIG_SCHEMA`, then
    converted to a :class:`~eradiate.util.collections.configdict`.
    """

    @classmethod
    @abstractmethod
    def config_schema(cls):
        """Cerberus validation schema to validate :data:`config`.

        See the `Cerberus documentation <https://docs.python-cerberus.org>`_
        for available rules.

        .. note::

            This member is implemented as an abstract property to force defining
            it upon subclassing :class:`ConfigObject`.
        """
        pass

    config = attr.ib(default={})

    def __attrs_post_init__(self):
        # Check config contents and normalise
        v = cerberus.Validator(self.config_schema())
        pint_unit = cerberus.TypeDefinition("pint_unit", (pint.unit.Unit,), ())
        v.types_mapping["pint_unit"] = pint_unit

        if not v.validate(self.config):
            raise ValueError(v.errors)

        self.config = configdict(v.normalized(self.config))

    def get_quantity(self, key):
        magnitude = self.config[key]
        unit = self.config.get(f"{key}_unit", None)
        if unit is None:
            return magnitude
        else:
            return ureg.Quantity(magnitude, unit)


config_default_units = DefaultUnits()
