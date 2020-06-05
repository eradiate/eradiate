"""Factory used to instantiate classes deriving from
:class:`~eradiate.scenes.base.SceneHelper` from dictionaries.
"""

import importlib
import inspect
import sys
from copy import deepcopy

from ..util.metaclasses import Singleton


class Factory(metaclass=Singleton):
    """This class implements an object factory which creates objects based on
    dictionaries. It is implemented as a singleton and looks up, upon
    instantiation, registered classes in modules listed in its
    :attr:`SUBMODULES` class attribute.

    A call to the constructor will always return the unique singleton instance.
    Upon initialisation, this instance will look up modules listed in the
    :attr:`SUBMODULES` and search for classes on which the :meth:`register`
    decorator has been applied. Note that decorated classes must implement
    a ``from_dict`` class method which generates instances from a dictionary.
    Since :class:`~eradiate.scenes.base.SceneHelper` has been designed to work
    with this class, it implements one.

    Using the factory simply requires to import it, instantiate it and call its
    :meth:`create` method with a dictionary.
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

            .. code:: python

                @attr.s
                @Factory.register("constant")
                class Constant(SceneHelper):
                    DEFAULT_CONFIG = {"radiance": 1.0}

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
        "atmosphere.rayleigh",
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
        return cls.from_dict(config_dict)
