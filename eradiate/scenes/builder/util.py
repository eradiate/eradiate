from functools import reduce

import attr
from attr.validators import instance_of as is_instance
from attr.validators import optional as is_optional

from ...util import always_iterable


def count(cls, lst):
    """Return the number of instances of a given class in a list."""
    return sum(isinstance(x, cls) for x in lst)


def load(obj):
    from eradiate.kernel.core.xml import load_string
    return load_string(obj.to_xml(add_version=True))


# Basic functional programming facilities
# [https://stackoverflow.com/questions/16739290/composing-functions-in-python]
def compose2(f, g):
    """
    Compose two functions from right to left
    """
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
    """
    Compose an arbitrary number of functions from right to left
    """
    return reduce(compose2, fs)


# Validators
def has_attribute_and_value(expected_attribute, expected_value):
    """
    Succeeds only if validated attribute has an expected attribute set to an
    expected value.
    """

    def f(instance, attribute, value):
        v = getattr(value, expected_attribute)
        if v is None:
            raise AttributeError(f"argument \'{attribute.name.strip('_')}\' "
                                 f"is missing attribute '{expected_attribute}'")
        if not v == expected_value:
            raise ValueError(f"wrong value for field '{expected_attribute}' "
                             f"[expected '{expected_value}', got '{v}']")

    return f


def has_length(expected_length):
    """
    Succeeds if validated attribute has the expected length.
    """

    expected_lengths = list(always_iterable(expected_length))

    def f(instance, attribute, value):
        l = len(value)
        if l not in expected_lengths:
            raise ValueError(f"field '{attribute.name}' has wrong length "
                             f"[expected one of {expected_lengths}, got {l}]")

    return f


# Converters
def convert_to(cls):
    """Convert to specified class (which must have a `convert` class method).
    Upon conversion failure, do not convert.
    """

    # Wrap non-tuple argument for convenience
    if not isinstance(cls, tuple):
        cls_tuple = (cls, )
    else:
        cls_tuple = cls

    # Check if target types all support conversion
    # (i.e. have a convert() class method)
    for _cls in cls_tuple:
        try:
            if not callable(_cls.convert):
                raise AttributeError
        except AttributeError:
            raise TypeError(
                f"target type {_cls} does not have a convert() class method")

    def f(x):
        for _cls in cls_tuple:
            try:
                return _cls.convert(x)
            except TypeError:
                continue
        # If all conversion attempts failed, do not convert
        return x

    return f


def set_name(value):
    """Set 'name' attribute of a given object."""

    def f(x):
        if hasattr(x, "name"):
            x.name = value
        return x

    return f


def seq_to_str(v, sep=", "):
    """Get string representation of object passed as an argument."""
    try:
        return sep.join(str(x) for x in v)
    except TypeError:
        return str(v)


# attrs helpers
def parameter(type, name=None, optional=True, default=None):
    """Define an attrs attribute representing a plugin parameter. This function
    can define a validator, converter and default value for the returned 
    attribute based on call arguments.

    :param (class or tuple) type: Expected parameter type. 
        Init value conversion is attempted using the :meth:`convert_to` 
        function.
        If a tuple of classes is passed, the init value's type must be of one of 
        them (see also :meth:`isinstance`). In that case, the converter attempts
        conversion to each of class until one works.
    :param (str or None) name: Parameter name. This will fill the parameter's
        `name` attribute automatically regardless that of the init value.
    :param (bool) optional: If `True`, make the parameter optional.
    :param default: Default value used if ``optional`` is `True` and no init 
        value is passed.

    :return (``attrs`` attribute): 
    """
    validator = is_instance(type)
    if optional:
        validator = is_optional(validator)

    converter_sequence = []
    if not isinstance(type, tuple):
        converter_sequence.insert(0, convert_to(type))
    else:
        converter_sequence.insert(0, convert_to(type[0]))
    if name is not None:
        converter_sequence.insert(0, set_name(name))

    if converter_sequence:
        converter = compose(*converter_sequence)
    else:
        converter = None

    return attr.ib(
        kw_only=True,
        default=default,
        validator=validator,
        converter=converter
    )
