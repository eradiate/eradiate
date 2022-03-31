from inspect import signature

import pytest

from eradiate.util.deprecation import DeprecatedWarning, UnsupportedWarning, deprecated


@pytest.mark.parametrize("comptype", ("function", "class"))
def test_deprecated(comptype):
    if comptype == "function":

        def component():
            ...

        wrapped = deprecated(deprecated_in="1.1")(component)
        # Decorator wraps function but preserves signature
        assert wrapped is not component
        assert signature(wrapped) == signature(component)

    elif comptype == "class":

        class component:
            ...

        component_new = component.__new__
        wrapped = deprecated(deprecated_in="1.1")(component)
        # Decorator only redefines __new__
        assert wrapped is component
        assert wrapped.__new__ is not component_new

    else:
        raise ValueError("unhandled case")

    wrapped = deprecated(deprecated_in="1.1", current_version="1.2")(component)
    with pytest.warns(DeprecatedWarning):
        wrapped()

    wrapped = deprecated(
        deprecated_in="1.1",
        removed_in="1.2",
        current_version="1.3",
    )(component)
    with pytest.warns(UnsupportedWarning):
        wrapped()
