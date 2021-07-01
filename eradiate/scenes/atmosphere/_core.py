from abc import ABC, abstractmethod
from typing import MutableMapping, Optional

import attr
import pinttr

from ..core import SceneElement
from ... import converters, validators
from ..._factory import BaseFactory
from ...attrs import AUTO, documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg


@parse_docs
@attr.s
class Atmosphere(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all atmospheres.
    """

    id = documented(
        attr.ib(
            default="atmosphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"atmosphere"',
    )

    toa_altitude = documented(
        pinttr.ib(
            default=AUTO,
            converter=converters.auto_or(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=validators.auto_or(
                pinttr.validators.has_compatible_units, validators.is_positive
            ),
            units=ucc.deferred("length"),
        ),
        doc="Altitude of the top-of-atmosphere level. If set to ``AUTO``, the "
        "TOA is inferred from the radiative properties profile provided it has "
        "one. Otherwise, a default value of 100 km is used.\n"
        "\n"
        "Unit-enabled field (default unit: cdu[length]).",
        type="float or AUTO",
        default="AUTO",
    )

    width = documented(
        pinttr.ib(
            default=AUTO,
            converter=converters.auto_or(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=validators.auto_or(
                pinttr.validators.has_compatible_units, validators.is_positive
            ),
            units=ucc.deferred("length"),
        ),
        doc="Atmosphere width. If set to ``AUTO``, a value will be estimated to "
        "ensure that the medium is optically thick. The implementation of "
        "this estimate depends on the concrete class inheriting from this "
        "one.\n"
        "\n"
        "Unit-enabled field (default unit: cdu[length]).",
        type="float or AUTO",
        default="AUTO",
    )

    def height(self):
        """
        Actual value of the atmosphere's height as a :class:`pint.Quantity`.
        If ``toa_altitude`` is set to ``AUTO``, a value of 100 km is returned;
        otherwise, ``toa_altitude`` is returned.
        """
        if self.toa_altitude is AUTO:
            return ureg.Quantity(100.0, ureg.km)
        else:
            return self.toa_altitude

    def kernel_height(self, ctx=None):
        """
        Return the height of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`pint.Quantity`:
            Height of the kernel object delimiting the atmosphere
        """
        return self.height() + self.kernel_offset(ctx)

    def kernel_offset(self, ctx=None):
        """
        Return vertical offset used to position the kernel object delimiting the
        atmosphere. The created cuboid shape will be shifted towards negative
        Z values by this amount.

        .. note::

           This is required to ensure that the surface is the only shape
           which can be intersected at ground level during ray tracing.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`pint.Quantity`:
            Vertical offset of cuboid shape.
        """
        return self.height() * 1e-3

    @abstractmethod
    def kernel_width(self, ctx=None):
        """
        Return width of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`pint.Quantity`:
            Width of the kernel object delimiting the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_phase(self, ctx=None):
        """
        Return phase function plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the phase
            functions attached to the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_media(self, ctx=None):
        """
        Return medium plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the media
            attached to the atmosphere.
        """
        pass

    @abstractmethod
    def kernel_shapes(self, ctx=None):
        """
        Return shape plugin specifications only.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            attached to the atmosphere.
        """
        pass

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        kernel_dict = {}

        if not ctx.ref:
            kernel_dict[self.id] = self.kernel_shapes(ctx=ctx)[f"shape_{self.id}"]
        else:
            kernel_dict[f"phase_{self.id}"] = self.kernel_phase(ctx=ctx)[
                f"phase_{self.id}"
            ]
            kernel_dict[f"medium_{self.id}"] = self.kernel_media(ctx=ctx)[
                f"medium_{self.id}"
            ]
            kernel_dict[f"{self.id}"] = self.kernel_shapes(ctx=ctx)[f"shape_{self.id}"]

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
