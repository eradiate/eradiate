"""Basic abstractions and utilities to assist with scene generation."""

from abc import ABC, abstractmethod
from collections import UserDict

import attr
import pinttr

import eradiate.kernel

from .._attrs import documented, parse_docs
from .._units import unit_registry as ureg
from ..exceptions import KernelVariantError


class KernelDict(UserDict):
    """A dictionary designed to contain a scene specification appropriate for
    instantiation with :func:`~mitsuba.core.xml.load_dict`.

    :class:`KernelDict` keeps track of the variant it has been created with
    and performs minimal checks to help prevent inconsistent scene creation.

    .. rubric:: Instance attributes

    ``data`` (dict):
        Wrapped dictionary.

    ``variant`` (str):
        Kernel variant for which the dictionary is created.
    """

    def __init__(self, *args, **kwargs):
        """Initialise self and set :attr:`variant` attribute based on currently
        set variant.

        Raises → :class:`~eradiate.util.exceptions.KernelVariantError`
            If no kernel variant is set.
        """
        variant = eradiate.kernel.variant()

        if variant is not None:
            #: Kernel variant for which the scene is created
            self.variant = variant
        else:
            raise KernelVariantError(
                "a kernel variant must be selected to create a KernelDict object"
            )

        super().__init__(*args, **kwargs)

    def check(self):
        """Perform basic checks on the dictionary:

        * check that the ``{"type": "scene"}`` parameter is included;
        * check if the variant for which the kernel dictionary was created is
          the same as the current one.

        Raises → ValueError
            If the ``{"type": "scene"}`` parameter is missing.

        Raises → :class:`.KernelVariantError`
            If the variant for which the kernel dictionary was created is
            not the same as the current one
        """
        variant = eradiate.kernel.variant()
        if self.variant != variant:
            raise KernelVariantError(
                f"scene dictionary created for kernel variant '{self.variant}', "
                f"incompatible with current variant '{variant}'"
            )

        if self.get("type", None) != "scene":
            raise ValueError(
                "kernel scene dictionary is missing {'type': 'scene'} parameters"
            )

    @classmethod
    def new(cls, *elements):
        """Create a kernel dictionary using the passed elements. This variadic
        function accepts an arbitrary number of positional arguments.

        Parameter ``elements`` (:class:`SceneElement` or dict):
            Items to add to the newly created kernel dictionary.

        Returns → :class:`KernelDict`
            Initialise kernel dictionary.
        """
        result = cls({"type": "scene"})
        result.add(*elements)
        return result

    def add(self, *elements):
        """Merge the content of a :class:`~eradiate.scenes.core.SceneElement` or
        another dictionary object with the current :class:`KernelDict`.

        Parameter ``elements`` (:class:`SceneElement` or dict):
            Items to add to the current kernel dictionary. If the item is a
            :class:`~eradiate.scenes.core.SceneElement` instance, its
            :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method will
            be called with ``ref`` set to ``True``. If it is a dictionary
            (including a :class:`.KernelDict`), it will be merged without change.
        """

        for element in elements:
            try:
                self.update(element.kernel_dict(ref=True))
            except AttributeError:
                self.update(element)

    def load(self):
        """Load kernel object from self.

        .. note:: Requires a valid selected operational mode.

        Returns → :class:`mitsuba.render.Scene`:
             Kernel object.
        """
        self.check()
        from eradiate.kernel.core.xml import load_dict

        return load_dict(self.data)


@parse_docs
@attr.s
class SceneElement(ABC):
    """Abstract class for all scene elements.

    This abstract base class provides a basic template for all scene element
    classes. It is implemented using the `attrs <https://www.attrs.org>`_ library.

    .. note::

        This class is designed to integrate with factory classes derived from
        :class:`.Factory`.
    """

    id = documented(
        attr.ib(
            default=None,
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="User-defined object identifier.",
        type="str or None",
        default="None",
    )

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary. This class method will additionally
        pre-process the passed dictionary to merge any field with an
        associated ``"_units"`` field into a :class:`pint.Quantity` container.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return cls(**d_copy)

    @abstractmethod
    def kernel_dict(self, ref=True):
        """Return a dictionary suitable for kernel scene configuration.

        Parameter ``ref`` (bool):
            If ``True``, use referencing for all relevant nested kernel plugins.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        pass
