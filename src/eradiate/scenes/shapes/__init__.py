from ._core import Shape, shape_factory
from ._cuboid import CuboidShape
from ._rectangle import RectangleShape
from ._sphere import SphereShape

__all__ = [
    "shape_factory",
    "CuboidShape",
    "Shape",
    "SphereShape",
    "RectangleShape",
]
