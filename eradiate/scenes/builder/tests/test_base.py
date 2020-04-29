import numpy as np
import pytest

from eradiate.scenes.builder.base import *


def test_object():
    class MyObject(Object):
        _tag = "obj"

    o = MyObject()
    assert o.to_xml() == """<obj/>"""
    assert o.to_xml(add_version=True) == """<obj version="0.1.0"/>"""

    o = MyObject(name="some_name")
    assert o.to_xml() == """<obj name="some_name"/>"""

    o = MyObject()
    o.name = "some_name"
    assert o.to_xml() == """<obj name="some_name"/>"""


def test_plugin(variant_scalar_mono):
    # Define concrete class
    class MyPlugin(Plugin):
        _tag = "bsdf"
        _type = "diffuse"

    # Create plugin interface object
    p = MyPlugin()

    # Check XML snippet
    assert p.to_xml() == """<bsdf type="diffuse"/>"""

    # Try and instantiate
    p_instance = p.instantiate()


def test_referable_plugin():
    class MyPlugin(ReferablePlugin):
        _tag = "bsdf"
        _type = "diffuse"

    p = MyPlugin(id="my_bsdf")
    assert p.to_xml() == """<bsdf type="diffuse" id="my_bsdf"/>"""
    assert p.get_ref().to_xml() == """<ref id="my_bsdf"/>"""


def test_ref():
    with pytest.raises(TypeError):
        Ref()

    with pytest.raises(TypeError):
        Ref(id=1.)

    r = Ref(id="some_id")
    assert r.to_xml() == """<ref id="some_id"/>"""


def test_bool():
    a = Bool(True)
    assert a.to_xml() == """<boolean value="true"/>"""
    a = Bool(False)
    assert a.to_xml() == """<boolean value="false"/>"""

    # Test converter
    b = Bool.convert(False)
    assert a == b and a is not b
    b = Bool.convert(a)
    assert a == b and a is not b


def test_int():
    i = Int(10)
    assert i.to_xml() == """<integer value="10"/>"""

    # Test converter
    j = Int.convert(10)
    assert i == j and i is not j
    j = Int.convert(i)
    assert i == j and i is not j


def test_float():
    f = Float(1.)
    assert f.to_xml() == """<float value="1.0"/>"""

    f = Float(name="some_float", value=1.)
    assert f.to_xml() == """<float name="some_float" value="1.0"/>"""

    # Test converter
    f = Float(1.)
    g = Float.convert(1.)
    assert f == g and f is not g
    g = Float.convert(f)
    assert f == g and f is not g


def test_point():
    p = Point()
    assert p.to_xml() == """<point value="0.0, 0.0, 0.0"/>"""

    with pytest.raises(ValueError):
        p = Point([1.0])

    p = Point([0, 0.5, 1])
    assert p.to_xml() == """<point value="0.0, 0.5, 1.0"/>"""

    # Test converter
    x = [0, 0.5, 1]
    p = Point(x)
    q = Point.convert(x)
    # Direct instance comparison is not possible, it's a known attrs issue
    # (see class definition)
    assert np.all(p.value == q.value) and p is not q


def test_vector():
    v = Vector(1.0)
    assert v.to_xml() == """<vector value="1.0"/>"""

    v = Vector([0, 0.5, 1])
    assert v.to_xml() == """<vector value="0.0, 0.5, 1.0"/>"""

    # Test converter
    for x in [1.0, [0, 0.5, 1]]:
        v = Vector(x)
        w = Vector.convert(x)
        # Direct instance comparison is not possible, it's a known attrs issue
        # (see class definition)
        assert np.all(v.value == w.value) and v is not w
