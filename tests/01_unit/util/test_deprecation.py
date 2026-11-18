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


def test_deprecated_kwargs():
    import attrs

    from eradiate.util.deprecation import deprecated_kwargs

    @deprecated_kwargs({"bar": None}, deprecated_in="1.1", removed_in="1.2")
    @attrs.define
    class Foo:
        foo: int

    @attrs.define
    class Sub(Foo):
        baz: int

    @deprecated_kwargs({"qux": ""}, deprecated_in="1.1", removed_in="1.2")
    @attrs.define(slots=False)
    class SubSub(Sub):
        pass

    @attrs.define(init=False)
    class NoInit(Foo):
        pass

    with pytest.warns(DeprecatedWarning, match="'Foo.bar' is deprecated"):
        assert Foo(foo=0, bar=1) == Foo(foo=0)

    with pytest.warns(DeprecatedWarning, match="'Sub.bar' is deprecated"):
        assert Sub(0, 1, bar=1) == Sub(0, 1)

    with pytest.warns(DeprecatedWarning) as record:
        assert SubSub(0, 1, bar=1, qux=1) == SubSub(0, 1)
    assert {str(w.message).split()[0] for w in record} == {
        "'SubSub.bar'",
        "'SubSub.qux'",
    }

    with pytest.warns(DeprecatedWarning) as record:
        NoInit.__init__(object.__new__(NoInit), 0, bar=1)
    assert len(record) == 1


def test_deprecated_kwargs_renamed():
    import attrs

    from eradiate.util.deprecation import deprecated_kwargs

    @deprecated_kwargs({"bar": "foo"}, deprecated_in="1.1", removed_in="1.2")
    @attrs.define
    class Foo:
        foo: int

    @attrs.define(slots=False)
    class Sub(Foo):
        baz: int

    with pytest.warns(DeprecatedWarning, match="'Sub.bar' is deprecated") as record:
        assert Sub(baz=1, bar=0) == Sub(0, 1)
    assert record[0].filename == __file__  # Warning points to the caller

    with pytest.raises(TypeError, match="cannot pass both"):
        Sub(foo=0, baz=1, bar=0)


def test_deprecated_kwargs_no_version():
    from eradiate.util.deprecation import deprecated_kwargs

    @deprecated_kwargs({"bar": None})
    class Foo:
        pass

    with pytest.warns(
        DeprecatedWarning, match="'Foo.bar' is deprecated. It is ignored."
    ):
        Foo(bar=1)
