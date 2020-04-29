import attr

from .base import Plugin


@attr.s
class ReconstructionFilter(Plugin):
    """
    Abstract base class for all reconstruction filter plugins.
    """
    _tag = "rfilter"


@attr.s
class Box(ReconstructionFilter):
    _type = "box"
