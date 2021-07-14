from abc import ABC, abstractmethod
from typing import MutableMapping, Optional, Union

import attr
import pint
import pinttr

from ..core import SceneElement
from ... import converters, validators
from ..._factory import Factory
from ...attrs import AUTO, documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...units import unit_context_config as ucc

atmosphere_factory = Factory()


@parse_docs
@attr.s
class Atmosphere(SceneElement, ABC):
    """
    An abstract base class defining common facilities for all atmospheres.

    An atmosphere consists of a kernel medium (with a phase function) attached
    to a kernel shape.

    .. note::
       The shape type is restricted to cuboid shapes at the moment.
    """

    id: Optional[str] = documented(
        attr.ib(
            default="atmosphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"atmosphere"',
    )

    width: Union[pint.Quantity, str] = documented(
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

    # --------------------------------------------------------------------------
    #                               Properties
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def bottom(self) -> pint.Quantity:
        """
        Return the atmosphere's bottom altitude.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's bottom altitude.
        """
        pass

    @property
    @abstractmethod
    def top(self) -> pint.Quantity:
        """
        Return the atmosphere's top altitude.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's top altitude.
        """
        pass

    @property
    def height(self) -> pint.Quantity:
        """
        Return the atmosphere's height.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's height.
        """
        return self.top - self.bottom

    # --------------------------------------------------------------------------
    #                           Evaluation methods
    # --------------------------------------------------------------------------

    @abstractmethod
    def eval_width(self, ctx: Optional[KernelDictContext] = None) -> pint.Quantity:
        """
        Return the Atmosphere's width.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Atmosphere's width.
        """
        pass

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @abstractmethod
    def kernel_phase(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
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
    def kernel_media(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
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
    def kernel_shapes(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
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

    def kernel_height(self, ctx: Optional[KernelDictContext] = None) -> pint.Quantity:
        """
        Return the height of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Height of the kernel object delimiting the atmosphere
        """
        return self.height + self.kernel_offset(ctx=ctx)

    def kernel_offset(self, ctx: Optional[KernelDictContext] = None) -> pint.Quantity:
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

        Returns → :class:`~pint.Quantity`:
            Vertical offset of cuboid shape.
        """
        return self.height * 1e-3

    def kernel_width(self, ctx: Optional[KernelDictContext] = None) -> pint.Quantity:
        """
        Return width of the kernel object delimiting the atmosphere.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → :class:`~pint.Quantity`:
            Width of the kernel object delimiting the atmosphere.
        """
        return self.eval_width(ctx=ctx)

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
            kernel_dict[self.id] = self.kernel_shapes(ctx=ctx)[f"shape_{self.id}"]

        return kernel_dict
