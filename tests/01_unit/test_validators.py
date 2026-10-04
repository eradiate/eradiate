import attrs
import pytest

from eradiate import unit_registry as ureg
from eradiate.validators import on_quantity


def test_on_quantity():
    v = on_quantity(attrs.validators.instance_of(float))

    # This should succeed
    v(None, None, 1.0)
    v(None, None, ureg.Quantity(1.0, "km"))

    # This should fail
    @attrs.define
    class Attribute:  # Tiny class to pass an appropriate attribute argument
        name = attrs.field()

    attribute = Attribute(name="attribute")

    with pytest.raises(TypeError):
        v(None, attribute, "1.")
