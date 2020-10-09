"""Basic abstractions and utilities to assist with scene generation."""

import importlib
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import attr

import eradiate.kernel
from ..util.attrs import attrib, unit_enabled
from ..util.exceptions import KernelVariantError
from ..util.units import ureg


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
        """Merge the content of a :class:`~eradiate.scenes.core.SceneHelper` or
        another dictionary object with the current :class:`KernelDict`.

        Parameter ``content`` (:class:`~eradiate.scenes.core.SceneHelper` or list or dict)
            Content to merge with the current scene. If ``content`` is a
            :class:`~eradiate.scenes.core.SceneHelper` instance, its
            :meth:`~eradiate.scenes.core.SceneHelper.kernel_dict` method will
            be called with ``ref`` set to `True`. If ``content`` is a list,
            :meth:`add` will be called for each element of it. If ``content`` is
            a dict, it will be merged without change.

        Returns → ``self``
        """
        # TODO: make variadic
        # TODO: accept merging KernelDict instances (after variant check)

        if isinstance(content, SceneHelper):
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
class SceneHelper(ABC):
    """Abstract class for all scene generation helpers.

    This abstract base class provides a basic template for all scene generation
    helper classes. It is implemented using the
    `attrs <https://www.attrs.org>`_ library.

    .. note::

        This class is designed to integrate with the :class:`SceneFactory` class.
        See the corresponding documentation for a list of factory-enabled
        scene generation helper classes.

    Constructor arguments / public attributes
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
        ``has_unit`` parameter. Unit-enabled fields are inspected.
        If a field ``field`` is a :class:`pint.Quantity`, it is
        converted to ``field_unit`` and replaced with its magnitude (it is
        stripped from its unit).

        From this, it follows that all unit-enabled fields are stored as their
        magnitude. The unit in which these magnitudes are stored are contained
        in the corresponding unit fields.
        """
        # TODO: transfer this to the unit_enabled decorator?

        # Strip units after checking consistency and converting
        for field in self._unit_enabled_field_names():
            value = getattr(self, field)
            if isinstance(value, ureg.Quantity):
                unit = getattr(self, f"{field}_unit")
                setattr(self, field, value.to(unit).magnitude)

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

        Returns → :class:`~eradiate.scenes.core.SceneHelper`:
            Created object.
        """
        return cls(**d)


class SceneHelperFactory:
    """This class implements an object factory which creates objects based on
    dictionaries. For optimal use, it needs to discover registered classes in
    modules listed in its :attr:`_submodules` class attribute.

    Class registration to this factory is done using the :meth:`register`
    declass decorator (which should be applied *after* the ``attr.s`` decorator.
    Note that decorated classes must implement a ``from_dict()`` class method
    which generates instances from a dictionary (all classes deriving from
    :class:`~eradiate.scenes.core.SceneHelper` do).

    Using the factory simply requires to import it and call its :meth:`create`
    class method with a dictionary containing:

    - a ``type`` key, whose value will be the name under which the class to
      be instantiated is registered in the factory;
    - dictionary contents which will be passed to the target class's
      ``from_dict()`` class method.

    .. note::

        This class is designed to allow for runtime instantiation and
        configuration of :class:`~eradiate.scenes.core.SceneHelper` child
        classes from YAML fragments.

    .. admonition:: Example

        The following code snippet instantiates a
        :class:`~eradiate.scenes.illumination.DirectionalIllumination` helper
        using its :factorykey:`directional` factory name:

        .. code:: python

            from eradiate.scenes.core import Factory

            illumination = Factory.create({
                "type": "directional",
                "irradiance": {"type": "uniform", "value": 1.0},
                "zenith": 30.0,
                "azimuth": 180.0
            })

        In practice, the ``type`` key is used to look up the class to
        instantiate, then popped from the configuration dictionary. Therefore,
        the corresponding object creation call is, in this particular case:

        .. code:: python

            DirectionalIllumination(
                irradiance={"type": "uniform", "value": 1.0},
                zenith=30.0,
                azimuth=180.0
            )

    .. admonition:: List of factory-enabled scene generation helpers
        :class: hint

        .. factorytable::
           :sections:
    """
    #: Internal registry for available scene helpers
    registry = {}

    #: List of submodules where to look for registered classes
    _submodules = [
        "atmosphere",
        "illumination",
        "lithosphere",
        "measure",
        "spectra"
    ]

    @classmethod
    def register(cls, name):
        """This decorator function is used on a class to register it to the
        factory.

        Parameter ``name`` (str):
            Name used to reference the registered class in the factory's
            registry.

        .. admonition:: Example
            :class: hint

            .. code:: python

                # Note that the register() decorator is applied *after* attr.s()
                @Factory.register(name="constant")
                @attr.s
                class ConstantIllumination(SceneHelper):
                    radiance = attr.ib(
                        validator=attr.validators.instance_of(float),
                        default=1.
                    )

                    def kernel_dict(self, **kwargs):
                        return {
                            self.id: {
                                "type": "constant",
                                "radiance": {"type": "uniform", "value": self.radiance}
                            }
                        }
        """

        def inner_wrapper(wrapped_class):

            if not hasattr(wrapped_class, "from_dict"):
                raise ValueError(f"class {wrapped_class} is missing 'from_dict()' "
                                 f"and cannot be registered")

            if name in cls.registry:
                warnings.warn(f"SceneHelper '{name}' already exists, will replace it")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, config_dict):
        """Create an instance from a class registered to the factory based on
        dictionary.

        Parameter ``config_dict`` (dict):
            Dictionary used to create an object. In addition to the parameters
            expected by the created object, it must have a 'type' entry
            containing the targeted object (lower-cased).

        Returns → :class:`SceneHelper`:
            Created object.

        Raises → ValueError:
            If ``type`` field does not map to a registered type.
        """

        # Retrieve relevant object from factory registry based on "type" key
        config_dict = deepcopy(config_dict)
        obj_type = config_dict.pop("type")

        try:
            return cls.registry[obj_type].from_dict(config_dict)
        except KeyError:
            raise ValueError(f"unregistered SceneHelper '{obj_type}'")

    @classmethod
    def convert(cls, value):
        """Object converter method.

        If ``value`` is a dictionary, this method forwards it to :meth:`create`.
        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.create(value)

        return value


# Import submodules containing classes to be automatically added to
# SceneHelperFactory's registry
for module_name in SceneHelperFactory._submodules:
    full_module_name = f"eradiate.scenes.{module_name}"
    importlib.import_module(full_module_name)
