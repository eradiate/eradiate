from abc import ABC, abstractmethod

import attr
import pinttr

from ..core import SceneElement
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory
from ..._units import unit_context_config as ucc


@parse_docs
@attr.s
class Canopy(SceneElement, ABC):
    """
    An abstract base class defining a base type for all canopies.
    """

    id = documented(
        attr.ib(
            default="canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"canopy"',
    )

    size = documented(
        pinttr.ib(
            default=None,
            validator=attr.validators.optional(
                [
                    pinttr.validators.has_compatible_units,
                    validators.on_quantity(validators.is_vector3),
                ]
            ),
            units=ucc.deferred("length"),
        ),
        doc="Canopy size as a 3-vector.\n\nUnit-enabled field (default: ucc[length]).",
        type="array-like",
    )

    @abstractmethod
    def bsdfs(self, ctx=None):
        """
        Return BSDF plugin specifications only.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the shapes in the canopy.
        """
        pass

    @abstractmethod
    def shapes(self, ctx=None):
        """
        Return shape plugin specifications only.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        pass

    @abstractmethod
    def kernel_dict(self, ctx=None):
        """
        Return a dictionary suitable for kernel scene configuration.

        Parameter ``ctx`` (:class:`.KernelDictContext` or None):
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        pass


class BiosphereFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`.Canopy`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: BiosphereFactory
    """

    _constructed_type = Canopy
    registry = {}
