import attr

from .base import Bool, Float, ReferablePlugin, String
from .spectra import Spectrum
from .textures import Texture
from .util import parameter


@attr.s
class BSDF(ReferablePlugin):
    """
    Abstract base class for all BSDF plugins.
    """
    _tag = "bsdf"


@attr.s
class Null(BSDF):
    """
    Null BSDF plugin.
    """
    _type = "null"
    _params = BSDF._params


_diffuse_params = {
    "reflectance": parameter(Spectrum, "reflectance")
}


@attr.s(these=_diffuse_params)
class Diffuse(BSDF):
    """
    Diffuse BSDF plugin.
    """
    _type = "diffuse"
    _params = BSDF._params + list(_diffuse_params.keys())


_rough_dielectric_params = {
    name: parameter(type, name)
    for name, type in [
        ("int_ior", (String, Float)),
        ("ext_ior", (String, Float)),
        ("specular_reflectance", (Spectrum, Texture)),
        ("specular_transmittance", (Spectrum, Texture)),
        ("distribution", String),
        ("alpha", (Float, Texture)),
        ("alpha_u", (Float, Texture)),
        ("alpha_v", (Float, Texture)),
        ("sample_visible", Bool)
    ]
}


@attr.s(these=_rough_dielectric_params)
class RoughDielectric(BSDF):
    """
    Rough dielectric BSDF plugin.
    """
    _type = "roughdielectric"
    _params = BSDF._params + list(_rough_dielectric_params.keys())
