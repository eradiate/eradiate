import attr

import eradiate

from .base import Int, Float, Plugin, Point, String, Vector
from .films import Film, HDRFilm
from .samplers import Sampler
from .transforms import Transform
from .util import parameter

_sensor_params = {
    name: parameter(type)
    for name, type in [
        ("sampler", Sampler),
        ("film", Film),
    ]
}


@attr.s(these=_sensor_params)
class Sensor(Plugin):
    """
    Abstract base class for all sensor plugins.
    """
    _tag = "sensor"
    _params = Plugin._params + list(_sensor_params.keys())

    def _ensure_film_size(self, width=1, height=1):
        """Fix wrong film settings"""

        if self.film is None:
            self.film = HDRFilm(width=width, height=height)

        if self.film.width is None or \
                self.film.height is None or \
                self.film.width.value != width or \
                self.film.height.value != height:

            from eradiate.kernel.core import Log, LogLevel
            Log(LogLevel.Warn,
                f"{type(self).__name__}: film size can only be {width}x{height}, applying fix")
            self.film.width = Int(name="width", value=width)
            self.film.height = Int(name="height", value=height)


_perspective_params = {
    name: parameter(type, name)
    for name, type in [
        ("to_world", Transform),
        ("fov", Float),
        ("fov_axis", String),
        ("near_clip", Float),
        ("far_clip", Float),
    ]
}


@attr.s(these=_perspective_params)
class Perspective(Sensor):
    """
    Perspective camera plugin.
    """
    _type = "perspective"
    _params = Sensor._params + list(_perspective_params.keys())


_radiancemeter_params = {
    name: parameter(type, name)
    for name, type in [
        ("origin", Point),
        ("direction", Vector),
    ]
}


@attr.s(these=_radiancemeter_params)
class RadianceMeter(Sensor):
    """
    Radiance meter plugin.
    """
    _type = "radiancemeter"
    _params = Sensor._params + list(_radiancemeter_params.keys())

    def __attrs_post_init__(self):
        self._ensure_film_size(width=1, height=1)


_distant_params = {
    name: parameter(type, name)
    for name, type in [
        ("direction", Vector),
        ("target", Point),
    ]
}


@attr.s(these=_distant_params)
class Distant(Sensor):
    """
    Radiance meter plugin.
    """
    _type = "distant"
    _params = Sensor._params + list(_distant_params.keys())

    def __attrs_post_init__(self):
        self._ensure_film_size(width=1, height=1)
