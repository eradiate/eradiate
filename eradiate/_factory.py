import warnings
from copy import deepcopy
from typing import Any, MutableMapping, Optional, Tuple, Type, Union

import attr
import dessinemoi


class BaseFactory:
    """Base class for factories.

    This class can be used to create object factories which instantiate
    objects based on dictionaries. Created factories handle only one type
    of object (and child classes), specified in a ``_constructed_type`` class
    attribute. By default, this member is set to ``object``: without any
    particular precaution, a factory deriving from this one accepts registration
    from any class, provided that it implements the required interface.

    Class registration to factory objects is done using the :meth:`register`
    class decorator (which should be applied *after* the :func:`attr.s` decorator).
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

    .. seealso::

       :ref:`sec-developer_guide-factory_guide`
           This guide gives an overview of the factory system and presents its
           usage.
    """

    #: By default, any object can be constructed by the factory
    _constructed_type = object

    #: Internal registry for available types
    registry = dict()

    @classmethod
    def register(cls, *names):
        """This decorator function is used on a class to register it to the
        factory. If the chosen registration name already exists in the factory's
        registry, this decorator will issue a warning.

        Parameter ``names`` (str):
            Names used to reference the registered class in the factory's
            registry. This parameter is variadic, meaning that multiple values
            can be passed to register and alias a class with multiple keys.

        Raises → TypeError:
            The decorated class cannot be registered because it is not supported
            by the factory.

        Raises → AttributeError:
            The decorated class cannot be registered because it is missing
            a ``from_dict()`` method.

        .. admonition:: Example
            :class: hint

            .. code:: python

                # Note that the register() decorator is applied **after** attr.s
                @IlluminationFactory.register("constant")
                @attr.s
                class ConstantIllumination(SceneElement):
                    ...
        """

        def inner_wrapper(wrapped_class):
            if not issubclass(wrapped_class, cls._constructed_type):
                raise TypeError(
                    f"cannot register class '{wrapped_class.__name__}' "
                    f"to {cls.__name__} (not a subclass of "
                    f"'{cls._constructed_type.__name__}')"
                )

            if not hasattr(wrapped_class, "from_dict"):
                raise AttributeError(
                    f"class '{wrapped_class.__name__}' is "
                    f"missing 'from_dict()' and cannot be "
                    f"registered"
                )

            for name in names:
                if name in cls.registry:
                    warnings.warn(
                        f"class '{wrapped_class.__name__}' already "
                        f"registered as '{name}', will be replaced"
                    )
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
            raise ValueError(
                f"no class registered as '{obj_type}'; "
                f"registered: {list(cls.registry.keys())}"
            )

    @classmethod
    def convert(cls, value):
        """Object converter method.

        If ``value`` is a dictionary, this method forwards it to :meth:`create`.
        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.create(value)

        return value


@attr.s
class Factory(dessinemoi.Factory):
    def _convert_impl(
        self,
        value,
        allowed_cls: Optional[Union[Type, Tuple[Type]]] = None,
    ) -> Any:
        if isinstance(value, MutableMapping):
            # Fetch class from registry
            type_id = value.pop("type")
            cls = self.registry[type_id]

            # Check if class is allowed
            if allowed_cls is not None and not issubclass(cls, allowed_cls):
                raise TypeError(
                    f"conversion to object type '{type_id}' ({cls}) is not allowed"
                )

            # Get constructor from dict, if any
            construct = value.pop("construct", None)

            # Construct object
            return self.create(type_id, construct=construct, kwargs=value)

        else:
            # Check if object has allowed type
            if allowed_cls is not None:
                if not isinstance(value, allowed_cls):
                    raise TypeError("value type is not allowed")

            return value
