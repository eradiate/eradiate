"""Basic facilities common to all atmosphere scene elements."""

from abc import ABC, abstractmethod

import attr

from ..core import SceneElement
from ...util.attrs import attrib, attrib_float_positive, attrib_units
from ...util.units import config_default_units as cdu
from ...util.units import ureg


def _converter_number_or_auto(value):
    if value == "auto":
        return value

    if isinstance(value, ureg.Quantity):
        return value

    return float(value)


def _validator_number_or_auto(_, attribute, value):
    if value == "auto":
        return

    if isinstance(value, ureg.Quantity):
        return

    if not isinstance(value, (int, float)):
        raise TypeError(f"{attribute.name} must be a 'float', 'int' or "
                        f"str('auto'), got {value} which is a '{type(value)}'")


@attr.s
class Atmosphere(SceneElement, ABC):
    """An abstract base class defining common facilities for all atmospheres.

    See :class:`~eradiate.scenes.core.SceneElement` for undocumented members.

    Constructor arguments / instance attributes:

        ``height`` (float):
            Atmosphere height. Default: 100.

            Unit-enabled field (default unit: cdu[length])

        ``width`` (float or "auto"):
            Atmosphere width. If set to ``"auto"``, a value will be estimated to
            ensure that the medium is optically thick. The implementation of
            this estimate depends on the concrete class inheriting from this
            one. Default: ``"auto"``.

            Unit-enabled field (default unit: cdu[length])
    """

    id = attrib(
        default="atmosphere",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    height = attrib_float_positive(
        default=1e2,
        has_units=True
    )
    height_units = attrib_units(
        compatible_units=ureg.m,
        default=attr.Factory(lambda: cdu.get("length"))
    )

    width = attrib(
        default="auto",
        converter=_converter_number_or_auto,
        validator=_validator_number_or_auto,
        has_units=True
    )
    width_units = attrib_units(
        compatible_units=ureg.m,
        default=attr.Factory(lambda: cdu.get("length")),
    )

    @property
    def _height(self):
        height = self.get_quantity("height")
        offset = height * 0.001  # TODO: maybe adjust offset based on medium profile
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
        kernel_dict = {"integrator": {"type": "volpath"}}  # Force volpath integrator

        if not ref:
            kernel_dict[self.id] = self.shapes()[f"shape_{self.id}"]
        else:
            kernel_dict[f"phase_{self.id}"] = self.phase()[f"phase_{self.id}"]
            kernel_dict[f"medium_{self.id}"] = self.media(ref=True)[f"medium_{self.id}"]
            kernel_dict[f"{self.id}"] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict
