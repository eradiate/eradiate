import warnings
from copy import deepcopy


class BaseFactory:
    """This class can be used to create object factories which instantiate
    objects based on dictionaries. Created factories handle only one type
    of object (and child classes), specified in a ``_constructed_type`` class
    attribute.

    Class registration to factory objects is done using the :meth:`register`
    class decorator (which should be applied *after* the ``attr.s`` decorator).
    Note that decorated classes must implement a ``from_dict()`` class method
    which generates instances from a dictionary. If a class with an unsupported
    type is decorated with :meth:`register`, a ``TypeError`` will be raised
    upon import.

    Using a factory created from this base class simply requires to import it
    and call its :meth:`create` class method with a dictionary containing:

    - a ``type`` key, whose value will be the name under which the class to
      be instantiated is registered in the factory;
    - dictionary contents which will be passed to the target class's
      ``from_dict()`` class method.

    .. note::

        This class is designed to allow for runtime instantiation and
        configuration of classes from YAML or JSON fragments.

    .. admonition:: Example

        The following code snippet instantiates a
        :class:`~eradiate.scenes.illumination.DirectionalIllumination` helper
        using its :factorykey:`directional` factory name:

        .. code:: python

            from eradiate.scenes.core import SceneElementFactory

            illumination = SceneElementFactory.create({
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
    """
    #: Internal registry for available types
    registry = {}

    @classmethod
    def register(cls, name):
        """This decorator function is used on a class to register it to the
        factory. If the chosen registration name already exists in the factory's
        registry, this decorator will issue a warning.

        Parameter ``name`` (str):
            Name used to reference the registered class in the factory's
            registry.

        Raises → TypeError:
            The decorated class cannot be registered because it is not supported
            by the factory.

         Raises → AttributeError:
            The decorated class cannot be registered because it is missing
            a ``from_dict()`` method.

        .. admonition:: Example
            :class: hint

            .. code:: python

                # Note that the register() decorator is applied *after* attr.s()
                @SceneElementFactory.register(name="constant")
                @attr.s
                class ConstantIllumination(SceneElement):
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
            if not issubclass(wrapped_class, cls._constructed_type):
                raise TypeError(f"cannot register class '{wrapped_class.__name__}' "
                                f"to {cls.__name__} (not a subclass of "
                                f"'{cls._constructed_type.__name__}')")

            if not hasattr(wrapped_class, "from_dict"):
                raise AttributeError(f"class '{wrapped_class.__name__}' is "
                                     f"missing 'from_dict()' and cannot be "
                                     f"registered")

            if name in cls.registry:
                warnings.warn(f"class '{wrapped_class.__name__}' already "
                              f"registered as '{name}', will be replaced")
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
            corresponding to name under which the desired object type is
            known in the registry.

        Returns → constructed type:
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
            raise ValueError(f"no class registered as '{obj_type}'")

    @classmethod
    def convert(cls, value):
        """Object converter method.

        If ``value`` is a dictionary, this method forwards it to :meth:`create`.
        Otherwise, it returns ``value``.

        .. note::

           This method is used to cascade object creation from nested
           dictionaries.
        """
        if isinstance(value, dict):
            return cls.create(value)

        return value
