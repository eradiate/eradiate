import attr

from .base import Float, ReferablePlugin
from .util import parameter


@attr.s
class PhaseFunction(ReferablePlugin):
    """
    Abstract base class for all phase function plugins.
    """
    _tag = "phase"


@attr.s
class Isotropic(PhaseFunction):
    """
    Isotropic phase function plugin.
    """
    _type = "isotropic"


_henyey_greenstein_params = {"g": parameter(Float, "g")}


@attr.s(these=_henyey_greenstein_params)
class HenyeyGreenstein(PhaseFunction):
    """
    Henyey-Greenstein phase function.
    """
    _type = "hg"
    _params = PhaseFunction._params + list(_henyey_greenstein_params.keys())


_rayleigh_params = {"delta": parameter(Float, "delta")}


@attr.s(these=_rayleigh_params)
class Rayleigh(PhaseFunction):
    """
    Rayleigh phase function.
    """
    _type = "rayleigh"
    _params = PhaseFunction._params + list(_rayleigh_params.keys())
