import pytest

from eradiate.scenes.builder.transforms import *


def test_translate():
    # Wrong init
    with pytest.raises(TypeError):
        Translate()

    # Init from positional argument
    t = Translate([0, 0, 1])
    assert t.to_xml() == """<translate value="0, 0, 1"/>"""


def test_rotate():
    # Wrong init
    with pytest.raises(TypeError):
        Rotate()

    # Wrong init
    with pytest.raises(ValueError):
        Rotate([0, 1], 90)

    # Init from positional argument
    t = Rotate([0, 0, 1], 90)
    assert t.to_xml() == """<rotate value="0, 0, 1" angle="90.0"/>"""


def test_scale():
    # Wrong init
    with pytest.raises(TypeError):
        Scale()

    # Init from float
    t = Scale(1)
    assert t.to_xml() == """<scale value="1"/>"""

    # Init from vector
    t = Scale([0, 0, 1])
    assert t.to_xml() == """<scale value="0, 0, 1"/>"""


def test_lookat():
    # Wrong init
    with pytest.raises(TypeError):
        LookAt()

    # Wrong init
    with pytest.raises(TypeError):
        LookAt([10, 50, -800], [0, 0, 0])

    # Init with positional arguments
    t = LookAt([10, 50, -800], [0, 0, 0], [0, 1, 0])
    assert t.to_xml() == """<lookat origin="10, 50, -800" target="0, 0, 0" up="0, 1, 0"/>"""

    # Init with kwargs
    t = LookAt(up=[0, 1, 0], origin=[10, 50, -800], target=[0, 0, 0])
    assert t.to_xml() == """<lookat origin="10, 50, -800" target="0, 0, 0" up="0, 1, 0"/>"""


def test_transform():
    # Init with empty sequence
    t = Transform()

    # Fail with empty sequence
    with pytest.raises(ValueError):
        t.to_etree()

    # Fill sequence
    t.sequence.append(Translate([0, 0, 1]))
    t.sequence.append(Rotate([0, 0, 1], 180))
    assert t.to_xml() == '<transform><translate value="0, 0, 1"/>' \
                         '<rotate value="0, 0, 1" angle="180.0"/></transform>'

    # Init with nonempty sequence
    t = Transform([
        Translate([1, 0, 0]),
        Rotate([0, 1, 0], 90),
        Scale(2),
        LookAt([10, 50, -800], [0, 0, 0], [0, 1, 0])
    ])
    assert t.to_xml() == '<transform>' \
                         '<translate value="1, 0, 0"/>' \
                         '<rotate value="0, 1, 0" angle="90.0"/>' \
                         '<scale value="2"/>' \
                         '<lookat origin="10, 50, -800" target="0, 0, 0" up="0, 1, 0"/>' \
                         '</transform>'
