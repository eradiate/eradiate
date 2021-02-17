import pint


def on_quantity(wrapped_converter):
    """Applies a converter to the magnitude of a :class:`pint.Quantity`."""

    def f(value):
        if isinstance(value, pint.Quantity):
            return wrapped_converter(value.magnitude) * value.units
        else:
            return wrapped_converter(value)

    return f


def auto_or(wrapped_converter):
    """Returns a converter which executes the wrapped converter if the converted
    value is not equal to ``"auto"``; otherwise returns ``"auto"``.
    """

    def f(value):
        if value == "auto":
            return value

        return wrapped_converter(value)

    return f
