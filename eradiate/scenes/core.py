"""Basic abstractions and utilities to assist with scene generation."""

import importlib
from abc import ABC, abstractmethod

import attr

import eradiate.kernel
from ..util.attrs import unit_enabled
from ..util.exceptions import KernelVariantError
from ..util.factory import BaseFactory


class KernelDict(dict):
    """A dictionary designed to contain a scene specification appropriate for
    instantiation with :func:`~mitsuba.core.xml.load_dict`.

    :class:`KernelDict` keeps track of the variant it has been created with
    and performs minimal checks to help prevent inconsistent scene creation.
    """

    # TODO: add a content() method which returns a generator with the content of
    #       the items() method without the 'type' entry

    def __init__(self, *args, **kwargs):
        """Initialise self and set :attr:`variant` attribute based on currently
        set variant.

        Raises → :class:`~eradiate.util.exceptions.KernelVariantError`
            If no kernel variant is set.
        """
        super(KernelDict, self).__init__(*args, **kwargs)

        variant = eradiate.kernel.variant()

        if variant is not None:
            self.variant = variant  #: Kernel variant with for which the scene is created
        else:
            raise KernelVariantError("a kernel variant must be selected to "
                                     "create a KernelDict object")

    @classmethod
    def empty(cls):
        """Create an empty scene."""
        return cls({"type": "scene"})

    def check(self):
        """Perform basic checks on the dictionary:

        - check that the ``{"type": "scene"}`` parameter is included;
        - check if the variant for which the kernel dictionary was created is
          the same as the current one.

        Raises → ValueError
            If  the ``{"type": "scene"}`` parameter is missing.

        Raises → :class:`.KernelVariantError`
            If the variant for which the kernel dictionary was created is
            not the same as the current one
        """
        if self.get("type", None) != "scene":
            raise ValueError("kernel scene dictionary is missing {'type': 'scene'} "
                             "parameters")

        variant = eradiate.kernel.variant()
        if self.variant != variant:
            raise KernelVariantError(f"scene dictionary created for kernel "
                                     f"variant '{self.variant}', incompatible "
                                     f"with current variant '{variant}'")

    def add(self, content):
        """Merge the content of a :class:`~eradiate.scenes.core.SceneElement` or
        another dictionary object with the current :class:`KernelDict`.

        Parameter ``content`` (:class:`~eradiate.scenes.core.SceneElement` or list or dict)
            Content to merge with the current scene. If ``content`` is a
            :class:`~eradiate.scenes.core.SceneElement` instance, its
            :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method will
            be called with ``ref`` set to `True`. If ``content`` is a list,
            :meth:`add` will be called for each element of it. If ``content`` is
            a dict, it will be merged without change.

        Returns → ``self``
        """
        # TODO: make variadic
        # TODO: accept merging KernelDict instances (after variant check)

        if isinstance(content, SceneElement):
            for key, value in content.kernel_dict(ref=True).items():
                self[key] = value
        elif isinstance(content, list):
            for item in content:
                self.add(item)
        else:
            for key, value in content.items():
                self[key] = value

        return self

    def normalize(self):
        self["type"] = "scene"

    def load(self):
        self.check()
        from eradiate.kernel.core.xml import load_dict
        return load_dict(self)


@unit_enabled
@attr.s
class SceneElement(ABC):
    """Abstract class for all scene elements.

    This abstract base class provides a basic template for all scene element
    classes. It is implemented using the `attrs <https://www.attrs.org>`_ library.

    .. note::

        This class is designed to integrate with factory classes derived from
        :class:`.Factory`.

    .. rubric:: Constructor arguments / instance attributes

    ``id`` (str or None):
        User-defined object identifier. Default: ``None``.
    """

    id = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    @abstractmethod
    def kernel_dict(self, ref=True):
        """Return a dictionary suitable for kernel scene configuration.

        Parameter ``ref`` (bool):
            If `True`, use referencing for all relevant nested kernel plugins.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        # TODO: return a KernelDict
        pass
