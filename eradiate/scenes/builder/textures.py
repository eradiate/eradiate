import attr

from .base import ReferablePlugin
from .spectra import Spectrum
from .transforms import Transform
from .util import parameter


_texture_params = {
    "to_uv": parameter(Transform, "to_uv"),
}


@attr.s(these=_texture_params)
class Texture(ReferablePlugin):
    """
    Abstract base class for all texture plugins.
    """
    _tag = "texture"
    _params = ReferablePlugin._params + list(_texture_params.keys())


_checkerboard_params = {
    name: parameter(type, name)
    for name, type in [
        ("color0", (Spectrum, Texture)),
        ("color1", (Spectrum, Texture)),
    ]
}


@attr.s(these=_checkerboard_params)
class Checkerboard(Texture):
    """
    Checkerboard texture plugin interface.
    """
    _type = "checkerboard"
    _params = Texture._params + list(_checkerboard_params.keys())