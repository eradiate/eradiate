import attr

from .base import Ref, ReferablePlugin
from .phase import PhaseFunction
from .spectra import Spectrum
from .util import parameter


@attr.s
class Medium(ReferablePlugin):
    """
    Abstract base class for all medium plugins.
    """
    _tag = "medium"


_homogeneous_params = {
    "phase": parameter((Ref, PhaseFunction)),
    "sigma_t": parameter(Spectrum, "sigma_t", default=Spectrum(1.0e-5)),
    "albedo": parameter(Spectrum, "albedo", default=Spectrum(0.99))
}


@attr.s(these=_homogeneous_params)
class Homogeneous(Medium):
    """
    Homogeneous medium plugin.
    """
    _type = "homogeneous"
    _params = Medium._params + list(_homogeneous_params.keys())

    @classmethod
    def from_collision_coefficients(cls, sigma_a, sigma_s, **kwargs):
        if "sigma_t" in kwargs.keys() or "albedo" in kwargs.keys():
            raise TypeError("sigma_a and sigma_s are exclusive with "
                            "sigma_t and albedo")
        sigma_t = sigma_a + sigma_s
        albedo = sigma_s / sigma_t
        return cls(sigma_t=sigma_t, albedo=albedo, **kwargs)

    @property
    def sigma_a(self):
        return self.sigma_t * (1. - self.albedo)

    @property
    def sigma_s(self):
        return self.sigma_t * self.albedo
