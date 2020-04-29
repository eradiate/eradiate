import attr

from .base import Plugin, Ref
from .bsdfs import BSDF
from .media import Medium
from .transforms import Transform
from .util import  parameter

_shape_params = {
    "to_world": parameter(Transform, "to_world"),
    "interior": parameter((Medium, Ref), "interior"),
    "exterior": parameter((Medium, Ref), "exterior")
}


@attr.s(these=_shape_params)
class Shape(Plugin):
    """
    Abstract base class for all shape plugins.
    """
    _tag = "shape"
    _params = Plugin._params + list(_shape_params.keys())


_rectangle_params = {"bsdf": parameter((Ref, BSDF))}


@attr.s(these=_rectangle_params)
class Rectangle(Shape):
    """
    Rectangle shape plugin.
    """
    _type = "rectangle"
    _params = Shape._params + list(_rectangle_params.keys())


_cube_params = {
    **_rectangle_params
}


@attr.s(these=_cube_params)
class Cube(Shape):
    """
    Cube shape plugin.
    """
    _type = "cube"
    _params = Shape._params + list(_cube_params.keys())
