import attr

from .base import Plugin, Vector
from .spectra import Spectrum
from .util import parameter


@attr.s
class Emitter(Plugin):
    """
   Abstract base class for all emitter plugins.
   """
    _tag = "emitter"


_constant_params = {"radiance": parameter(Spectrum, "radiance")}


@attr.s(these=_constant_params)
class Constant(Emitter):
    """
    Constant environment source plugin.
    """
    _type = "constant"
    _params = Emitter._params + list(_constant_params.keys())


_distant_params = {
    name: parameter(type, name, default=default)
    for name, type, default in [
        ("direction", Vector, None),
        ("irradiance", Spectrum, Spectrum(1.0)),
    ]
}


@attr.s(these=_distant_params)
class Directional(Emitter):
    """
    Distant directional emitter plugin.
    """
    _type = "directional"
    _params = Emitter._params + list(_distant_params.keys())


_area_params = {
    "radiance": parameter(Spectrum, "radiance", default=Spectrum(1.0))
}


@attr.s(these=_area_params)
class Area(Emitter):
    """
    Area emitter plugin.
    """
    _type = "area"
    _params = Emitter._params + list(_area_params.keys())
