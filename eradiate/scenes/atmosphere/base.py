"""Basic facilities common to all atmosphere scene elements."""

from abc import ABC, abstractmethod

import attr

from ..core import SceneElement
from ...util.attrs import (
    attrib_quantity, converter_to_units, validator_is_positive,
    validator_units_compatible
)
from ...util.units import config_default_units as cdu
from ...util.units import ureg


def _converter_or_auto(wrapped_converter):
    def f(value):
        if value == "auto":
            return value

        return wrapped_converter(value)

    return f


def _validator_or_auto(wrapped_validator):
    def f(instance, attribute, value):
        if value == "auto":
            return

    return f


@attr.s
class Atmosphere(SceneElement, ABC):
    """An abstract base class defining common facilities for all atmospheres.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``height`` (float):
        Atmosphere height. Default: 100 km.

        Unit-enabled field (default unit: cdu[length])

    ``width`` (float or "auto"):
        Atmosphere width. If set to ``"auto"``, a value will be estimated to
        ensure that the medium is optically thick. The implementation of
        this estimate depends on the concrete class inheriting from this
        one. Default: ``"auto"``.

        Unit-enabled field (default unit: cdu[length])
    """

    id = attr.ib(
        default="atmosphere",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    height = attrib_quantity(
        default=ureg.Quantity(100., ureg.km),
        validator=validator_is_positive,
        units_compatible=cdu.generator("length"),
    )

    width = attrib_quantity(
        default="auto",
        converter=_converter_or_auto(converter_to_units(cdu.generator("length"))),
        validator=_validator_or_auto(validator_units_compatible(ureg.m)),
        units_compatible=ureg.m,
        units_add_converter=False,
        units_add_validator=False
    )

    @property
    def _height(self):
        height = self.height
        offset = height * 0.001  # TODO: adjust offset based on medium profile
        return height, offset

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
        # TODO: extract integrator setup
        kernel_dict = {"integrator": {"type": "volpath"}}  # Force volpath integrator

        if not ref:
            kernel_dict[self.id] = self.shapes()[f"shape_{self.id}"]
        else:
            kernel_dict[f"phase_{self.id}"] = self.phase()[f"phase_{self.id}"]
            kernel_dict[f"medium_{self.id}"] = self.media(ref=True)[f"medium_{self.id}"]
            kernel_dict[f"{self.id}"] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict
