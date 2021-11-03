import typing as t

import pint

from .attrs import AUTO


def on_quantity(
    wrapped_converter: t.Callable[[t.Any], t.Any]
) -> t.Callable[[t.Any], t.Any]:
    """
    Apply a converter to the magnitude of a :class:`pint.Quantity`.

    Parameters
    ----------
    wrapped_converter : callable
        The converter which will be applied to the magnitude of a
        :class:`pint.Quantity`.

    Returns
    -------
    callable
    """

    def f(value: t.Any) -> t.Any:
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def auto_or(
    wrapped_converter: t.Callable[[t.Any], t.Any]
) -> t.Callable[[t.Any], t.Any]:
    """
    A converter that allows an attribute to be set to :data:`.AUTO`.

    Parameters
    ----------
    wrapped_converter : callable
        The converter that is used for non-:data:`.AUTO` values.

    Returns
    -------
    callable
    """

    def f(value):
        if value is AUTO:
            return value

        return wrapped_converter(value)

    return f
