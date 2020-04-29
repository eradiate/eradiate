import attr

from .base import Int, Plugin
from .util import parameter


@attr.s
class Integrator(Plugin):
    """
    Abstract base class for all integrator plugins.
    """
    _tag = "integrator"


_direct_params = {
    name: parameter(type, name)
    for name, type in [
        ("shading_samples", Int),
        ("emitter_samples", Int),
        ("bsdf_samples", Int),
        # ("hide_emitters", Bool)  # Not implemented at the moment
    ]
}


@attr.s(these=_direct_params)
class Direct(Integrator):
    """
    Direct illumination integrator plugin.
    """
    _type = "direct"
    _params = Integrator._params + list(_direct_params.keys())


_path_params = {
    name: parameter(type, name)
    for name, type in [
        ("max_depth", Int),
        ("rr_depth", Int),
        # ("hide_emitters", Bool),  # Not implemented at the moment
    ]
}


@attr.s(these=_path_params)
class Path(Integrator):
    """
    Path tracer integrator plugin.
    """
    _type = "path"
    _params = Integrator._params + list(_path_params.keys())


_vol_path_params = {
    "max_depth": parameter(Int, "max_depth")
}


@attr.s(these=_vol_path_params)
class VolPath(Integrator):
    """
    Volumetric path tracer plugin.
    """
    _type = "volpath"
    _params = Integrator._params + list(_vol_path_params.keys())
