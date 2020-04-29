import attr

from .base import Bool, Int, Plugin, String
from .rfilters import ReconstructionFilter
from .util import parameter


@attr.s
class Film(Plugin):
    """
    Abstract base class for all film plugins.
    """
    _tag = "film"


_hdrfilm_params = {
    **{
        name: parameter(type, name)
        for name, type in [
            ("width", Int),
            ("height", Int),
            ("file_format", String),
            ("pixel_format", String),
            ("component_format", String),
            ("crop_offset_x", Int),
            ("crop_offset_y", Int),
            ("crop_width", Int),
            ("crop_height", Int),
            ("high_quality_edges", Bool),
        ]
    },
    **{"rfilter": parameter(type=ReconstructionFilter)}
}


@attr.s(these=_hdrfilm_params)
class HDRFilm(Film):
    """
    High-dynamic range film plugin.
    """
    _type = "hdrfilm"
    _params = Film._params + list(_hdrfilm_params.keys())
