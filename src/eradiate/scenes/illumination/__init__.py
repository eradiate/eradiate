from ._constant import ConstantIllumination
from ._core import Illumination, illumination_factory
from ._directional import DirectionalIllumination

__all__ = [
    "Illumination",
    "illumination_factory",
    "ConstantIllumination",
    "DirectionalIllumination",
]
