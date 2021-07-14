import attr

from eradiate._factory import Factory
from eradiate.units import unit_registry as ureg


def test_factory_convert():
    factory = Factory()

    # We register a type with a nondefault constructor used for dict-based
    # creation
    @factory.register(type_id="mycls", dict_constructor="foo")
    @attr.s
    class MyClass:
        field = attr.ib(default=None)

        @classmethod
        def foo(cls):
            return cls(field="foo")

        @classmethod
        def bar(cls):
            return cls(field="bar")

    # MyClass instances are not modified
    o = MyClass()
    assert factory.convert(o) is o

    # Dict conversion uses the specified constructor
    assert factory.convert({"type": "mycls"}) == MyClass(field="foo")

    # We can override the dict constructor, either to use the default one...
    assert factory.convert({"type": "mycls", "construct": None}) == MyClass()
    # ... or a custom one
    assert factory.convert({"type": "mycls", "construct": "bar"}) == MyClass(
        field="bar"
    )

    # Units are attached to fields prior to conversion
    assert factory.convert(
        {"type": "mycls", "construct": None, "field": 1.0, "field_units": "m"}
    ) == MyClass(field=1.0 * ureg.m)
