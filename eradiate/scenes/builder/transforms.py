import attr
import numpy as np
from attr.validators import deep_iterable, instance_of

from .base import Object
from .util import has_length, seq_to_str
from ...util import ensure_array



__all__ = [
    "Transform",
    "Translate",
    "Rotate",
    "Scale",
    "LookAt"
]


@attr.s
class Translate(Object):
    """
    Translation transformation.
    """

    _tag = "translate"
    # Translation vector
    value = attr.ib(converter=np.array, validator=has_length(3))

    def to_etree(self):
        e = super().to_etree()
        e.set("value", seq_to_str(self.value))
        return e


@attr.s
class Rotate(Object):
    """
    Rotation transformation.
    """
    _tag = "rotate"
    value = attr.ib(converter=np.array,
                    validator=has_length(3))  # Rotation axis
    angle = attr.ib(converter=float)  # Rotation angle (in degrees)

    def to_etree(self):
        e = super().to_etree()
        e.set("value", seq_to_str(self.value))
        e.set("angle", str(self.angle))
        return e


@attr.s
class Scale(Object):
    """
    Scaling transformation.
    """
    _tag = "scale"
    value = attr.ib(converter=ensure_array, validator=has_length(
        [1, 3]))  # Scaling factor or vector

    def to_etree(self):
        e = super().to_etree()
        e.set("value", seq_to_str(self.value))
        return e


@attr.s
class LookAt(Object):
    """
    Look at transformation.
    """
    _tag = "lookat"
    origin = attr.ib(converter=np.array, validator=has_length(3))
    target = attr.ib(converter=np.array, validator=has_length(3))
    up = attr.ib(converter=np.array, validator=has_length(3))

    def to_etree(self):
        e = super().to_etree()
        e.set("origin", seq_to_str(self.origin))
        e.set("target", seq_to_str(self.target))
        e.set("up", seq_to_str(self.up))
        return e


@attr.s
class Transform(Object):
    """
    Transformation sequence.
    """
    _tag = "transform"
    sequence = attr.ib(
        default=[],
        validator=deep_iterable(
            member_validator=instance_of((Translate, Rotate, Scale, LookAt))
        ),
        converter=list,
    )  # Transformation sequence (elements can be any of the different transformation types)

    def to_etree(self):
        e = super().to_etree()
        if not self.sequence:
            raise ValueError("empty transform sequence")
        for x in self.sequence:
            e.append(x.to_etree())
        return e
