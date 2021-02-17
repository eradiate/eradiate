"""Basic facilities common to all atmosphere scene elements."""

from abc import (
    ABC,
    abstractmethod
)

import attr
import pinttr

from ..core import SceneElement
from ... import (
    converters,
    validators
)
from ..._attrs import (
    documented,
    get_doc,
    parse_docs
)
from ..._factory import BaseFactory
from ..._units import unit_context_config as ucc
from ..._units import unit_registry as ureg


@parse_docs
@attr.s
class Atmosphere(SceneElement, ABC):
    """An abstract base class defining common facilities for all atmospheres.
    """

    id = documented(
        attr.ib(
            default="atmosphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default="\"atmosphere\""
    )

    toa_altitude = documented(
        pinttr.ib(
            default="auto",
            converter=converters.auto_or(pinttr.converters.to_units(
                ucc.deferred("length")
            )),
            validator=validators.auto_or(
                pinttr.validators.has_compatible_units,
                validators.is_positive
            ),
            units=ucc.deferred("length"),
        ),
        doc="Altitude of the top-of-atmosphere level. If set to ``\"auto\"``, the "
            "TOA is inferred from the radiative properties profile provided it has "
            "one. Otherwise, a default value of 100 km is used.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[length]).",
        type="float or \"auto\"",
        default="\"auto\"",
    )

    width = documented(
        pinttr.ib(
            default="auto",
            converter=converters.auto_or(pinttr.converters.to_units(
                ucc.deferred("length")
            )),
            validator=validators.auto_or(
                pinttr.validators.has_compatible_units,
                validators.is_positive
            ),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere width. If set to ``\"auto\"``, a value will be estimated to "
            "ensure that the medium is optically thick. The implementation of "
            "this estimate depends on the concrete class inheriting from this "
            "one.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[length]).",
        type="float or \"auto\"",
        default="\"auto\"",
    )

    @property
    def height(self):
        """Actual value of the atmosphere's height as a :class:`pint.Quantity`.
        If ``toa_altitude`` is set to ``"auto"``, a value of 100 km is returned;
        otherwise, ``toa_altitude`` is returned.
        """
        if self.toa_altitude == "auto":
            return ureg.Quantity(100., ureg.km)
        else:
            return self.toa_altitude

    @property
    def kernel_height(self):
        """Height of the kernel object delimiting the atmosphere."""
        return self.height + self.kernel_offset

    @property
    def kernel_offset(self):
        """Vertical offset used to position the kernel object delimiting the
        atmosphere. The created cuboid shape will be shifted towards negative
        Z values by this amount.

        .. note::

           This is required to ensure that the surface is the only shape
           which can be intersected at ground level during ray tracing.
        """
        return self.height * 1e-3

    @property
    @abstractmethod
    def kernel_width(self):
        """Width of the kernel object delimiting the atmosphere."""
        pass

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
        kernel_dict = {}

        if not ref:
            kernel_dict[self.id] = self.shapes()[f"shape_{self.id}"]
        else:
            kernel_dict[f"phase_{self.id}"] = self.phase()[f"phase_{self.id}"]
            kernel_dict[f"medium_{self.id}"] = self.media(ref=True)[f"medium_{self.id}"]
            kernel_dict[f"{self.id}"] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict


class AtmosphereFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`.Atmosphere`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: AtmosphereFactory
    """
    _constructed_type = Atmosphere
    registry = {}
