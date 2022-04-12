import pytest

from eradiate.util.deprecation import DeprecatedWarning, UnsupportedWarning, deprecated


@pytest.mark.parametrize(
    "deprecation_type, expected_warning",
    (("deprecated", DeprecatedWarning), ("unsupported", UnsupportedWarning)),
)
def test_deprecated(deprecation_type, expected_warning):
    if deprecation_type == "deprecated":
        wrapper = deprecated(deprecated_in="1.1", current_version="1.2")

    elif deprecation_type == "unsupported":
        wrapper = deprecated(
            deprecated_in="1.1", removed_in="1.2", current_version="1.3"
        )

    else:
        raise ValueError("unhandled case")

    @wrapper
    def function():
        pass

    @wrapper
    class Class:
        @wrapper
        def instancemethod(self):
            pass

        @classmethod
        @wrapper
        def classmethod(cls):
            pass

        @staticmethod
        @wrapper
        def staticmethod():
            pass

    with pytest.warns(expected_warning):
        function()

    with pytest.warns(expected_warning):
        obj = Class()

    with pytest.warns(expected_warning):
        obj.instancemethod()

    with pytest.warns(expected_warning):
        Class.classmethod()

    with pytest.warns(expected_warning):
        Class.staticmethod()
