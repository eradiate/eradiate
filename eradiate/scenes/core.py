"""Basic abstractions and utilities to assist with scene generation."""

import importlib
import inspect
import sys
from abc import abstractmethod
from copy import deepcopy

import attr

import eradiate.kernel
from ..util.config_object import ConfigObject
from ..util.exceptions import KernelVariantError
from ..util.metaclasses import Singleton


@attr.s
class SceneHelper(ConfigObject):
    """Abstract class for all scene generation helpers.

    This abstract base class provides a basic template for all scene generation
    helper classes. Since it inherits from
    :class:`~eradiate.scenes.core.ConfigObject`, its subclasses must define
    a :data:`CONFIG_SCHEMA` class attribute.

    .. note::

        This class is designed to integrate with the :class:`Factory` class.
        See the corresponding documentation for a list of factory-enabled
        scene generation helper classes.

    Instance attributes:
        ``id`` (str)
            Unique identifier used to prefix kernel dictionary objects.
    """

    id = attr.ib(default=None)  #: Base identifier

    def __attrs_post_init__(self):
        super(SceneHelper, self).__attrs_post_init__()

        # Initialise internal state
        self.init()

    def init(self):
        """(Re)initialise internal state.

        This method is automatically called by the constructor to initialise the
        object."""
        pass

    @abstractmethod
    def kernel_dict(self, ref=True):
        """Return dictionary suitable for kernel scene configuration.

        Parameter ``ref`` (bool):
            If `True`, use referencing for all relevant nested plugins.

        Returns → dict:
            Dictionary suitable for merge with a kernel scene dictionary
            (using :func:`~mitsuba.core.xml.load_dict`).
        """
        # TODO: return a KernelDict
        pass

    @classmethod
    def from_dict(cls, d):
        """Create from a configuration dictionary.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation. The configuration
            dictionary uses the same structure as :data:`DEFAULT_CONFIG`.

        Returns → :class:`~eradiate.scenes.core.SceneHelper`:
            Created object.
        """
        return cls(config=d)


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


class Factory(metaclass=Singleton):
    """This class implements an object factory which creates objects based on
    dictionaries. It is implemented as a singleton and looks up, upon
    instantiation, registered classes in modules listed in its
    :attr:`SUBMODULES` class attribute.

    A call to the constructor will always return the unique singleton instance.
    Upon initialisation, this instance will look up modules listed in the
    :attr:`SUBMODULES` and search for classes on which the :meth:`register`
    decorator has been applied. Note that decorated classes must implement
    a ``from_dict()`` class method which generates instances from a dictionary.
    Since :class:`~eradiate.scenes.core.SceneHelper` has been designed to work
    with this class, it implements one.

    Using the factory simply requires to import it, instantiate it and call its
    :meth:`create` method with a dictionary containing:

    - a ``type`` key, whose value will be the keyword corresponding to the
      object the factory will create;
    - dictionary contents which will be passed to the target class's
      ``from_dict()`` class method.

    Eradiate only registers :class:`~eradiate.scenes.core.SceneHelper`
    derivatives to its factory by default. Consequently, configuration
    dictionary validation will occur upon calling :meth:`create`.

    .. note::

        This class is designed to allow for runtime instantiation and
        configuration of :class:`~eradiate.scenes.core.SceneHelper` children
        classes from YAML fragments.


    .. admonition:: Example

        The following code snippet instantiates a
        :class:`~eradiate.scenes.illumination.DistantIllumination` helper using
        its :factorykey:`distant` factory key:

        .. code:: python

            from eradiate.scenes.core import Factory

            factory = Factory()
            illumination = factory.create({
                "type": "distant",
                "irradiance": 1.0,
                "zenith": 30.0,
                "azimuth": 180.0
            })

        In practice, the ``type`` key is used to look up the class to
        instantiate, then popped from the configuration dictionary. Therefore,
        the corresponding object creation call is, in this particular case:

        .. code:: python

            DistantSurface({
                "irradiance": 1.0,
                "zenith": 30.0,
                "azimuth": 180.0
            })

    .. admonition:: List of factory-enabled scene generation helpers
        :class: hint

        .. factorytable::
           :sections:
    """

    @staticmethod
    def register(name=None):
        """This decorator function is used on a class to register it to the
        factory.

        Parameter ``name`` (str):
            If set, string used to reference the registered class in the
            factory's table. If unset, this name will default to the registered
            classe's name converted to lower case.

        .. admonition:: Example
            :class: hint

            .. code:: python

                @attr.s
                @Factory.register(name="constant")
                class ConstantIllumination(SceneHelper):
                    CONFIG_SCHEMA = frozendict({
                        "radiance": {
                            "type": "number",
                            "min": 0.,
                            "default": 1.
                        }
                    })

                    id = attr.ib(default="illumination")

                    def kernel_dict(self, **kwargs):
                        radiance = self.config["radiance"]
                        return {
                            self.id: {
                                "type": "constant",
                                "radiance":
                                    {"type": "uniform", "value": radiance}
                                    if isinstance(radiance, float)
                                    else radiance
                            }
                        }
        """

        def deco(cls):
            if not hasattr(cls, "from_dict"):
                raise ValueError(f"class {cls} is missing 'from_dict()' and "
                                 f"cannot be registered")
            if name is None:
                setattr(cls, "_REGISTERED", cls.__name__.lower())
            else:
                setattr(cls, "_REGISTERED", name)
            return cls

        return deco

    SUBMODULES = [
        "atmosphere",
        "illumination",
        "lithosphere",
        "measure"
    ]  #: List of submodules where to look for registered classes

    def __init__(self):
        # Initialise factory dict
        self.table = {}  #: Dictionary listing ``type`` keywords and the associated class

        # Discover supported classes in relevant submodules
        for module_name in self.SUBMODULES:
            full_module_name = f"eradiate.scenes.{module_name}"
            importlib.import_module(full_module_name)
            cls_members = inspect.getmembers(
                sys.modules[full_module_name], inspect.isclass
            )

            for cls_name, cls in cls_members:
                if hasattr(cls, "_REGISTERED"):
                    if hasattr(cls, "from_dict"):
                        self.table[cls._REGISTERED] = cls
                    else:
                        raise AttributeError(
                            f"class {full_module_name}.{cls_name} is "
                            f"registered to the factory but misses "
                            f"'from_dict()' class method"
                        )

    def create(self, config_dict):
        """Create an instance from a class registered to the factory's table
        based on dictionary.

        Parameter ``config_dict`` (dict):
            Dictionary used to create an object. In addition to the parameters
            expected by the created object, it must have a 'type' entry
            containing the targeted object (lower-cased).
        """
        # Retrieve relevant object from factory dict based on "type" key
        obj_type = config_dict["type"]

        try:
            cls = self.table[obj_type]
        except KeyError:
            raise ValueError(f"unregistered object type '{obj_type}'")

        # Instantiate object and return it
        config_dict = deepcopy(config_dict)
        del (config_dict["type"])
        return cls(config_dict)
