"""Basic abstractions and utilities to assist with scene generation."""

import importlib
from abc import ABC, abstractmethod

import attr

import eradiate.kernel
from ..util.attrs import attrib, unit_enabled
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

        This class is designed to integrate with the :class:`SceneElementFactory`
        class. See the corresponding documentation for a list of factory-enabled
        scene element classes.

    .. rubric:: Constructor arguments / instance attributes

    ``id`` (str or None):
        User-defined object identifier. Default: ``None``.
    """

    id = attrib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    def __attrs_post_init__(self):
        """This post-init step handles unit-enabled fields.
        Fields can be unit-enabled using :func:`~eradiate.util.attrs.attrib`'s
        ``has_units`` parameter. Unit-enabled fields are inspected.
        If a field ``field`` is a :class:`pint.Quantity`, it is
        converted to ``field_units`` and replaced with its magnitude (it is
        stripped from its unit).

        From this, it follows that all unit-enabled fields are stored as their
        magnitude. The unit in which these magnitudes are stored are contained
        in the corresponding unit fields.
        """
        self._strip_units()
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

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → :class:`~eradiate.scenes.core.SceneElement`:
            Created object.
        """
        return cls(**d)


class SceneElementFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`SceneElement`. For optimal use, it needs to discover registered
    classes in modules listed in its :attr:`_modules` class attribute.

    .. admonition:: Registered factory members
        :class: hint

        .. factorytable::
           :factory: SceneElementFactory
           :sections:
    """

    _constructed_type = SceneElement
    registry = {}

    #: List of submodules where to look for registered classes
    _modules = [
        "eradiate.scenes.atmosphere",
        "eradiate.scenes.illumination",
        "eradiate.scenes.lithosphere",
        "eradiate.scenes.measure",
        "eradiate.scenes.spectra"
    ]

    @classmethod
    def _discover(cls):
        """Import submodules containing classes to be automatically added to
        :class:`SceneElementFactory`'s registry."""
        for module_name in cls._modules:
            importlib.import_module(module_name)


# Trigger factory module discovery
SceneElementFactory._discover()
