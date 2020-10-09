"""Homogeneous atmosphere scene elements."""

import attr

import eradiate
from .base import Atmosphere, _validator_number_or_auto
from .radiative_properties.rayleigh import sigma_s_air
from ..core import SceneElementFactory
from ...util.attrs import attrib, attrib_unit
from ...util.units import config_default_units as cdu
from ...util.units import kernel_default_units as kdu
from ...util.units import ureg


@SceneElementFactory.register("rayleigh_homogeneous")
@attr.s()
class RayleighHomogeneousAtmosphere(Atmosphere):
    """Rayleigh homogeneous atmosphere scene element
    [:factorykey:`rayleigh_homogeneous`].

    This class builds an atmosphere consisting of a non-absorbing
    homogeneous medium. Scattering uses the Rayleigh phase function and the
    Rayleigh scattering coefficient of air at standard number density (
    see :func:`sigma_s_air`).

    See :class:`~eradiate.scenes.atmosphere.base.Atmosphere` for undocumented
    members.

    Constructor arguments / instance attributes:
        ``sigma_s`` (float or "auto"):
            Atmosphere scattering coefficient value. If set to ``"auto"``,
            the scattering coefficient will be computed based on the current
            operational mode configuration using the :func:`sigma_s_air`
            function. Default: ``"auto"``.


            Unit-enabled field (default unit: cdu[length]^-1).

    """

    sigma_s = attrib(
        default="auto",
        converter=lambda x: x if x == "auto" else float(x),
        validator=_validator_number_or_auto,
        has_unit=True
    )
    sigma_s_unit = attrib_unit(
        default=attr.Factory(lambda: cdu.get("length") ** -1),
        compatible_units=ureg.m ** -1,
    )

    @property
    def _albedo(self):
        """Return albedo."""
        return 1.

    @property
    def _sigma_s(self):
        """Return scattering coefficient based on configuration."""
        if self.sigma_s == "auto":
            wavelength = eradiate.mode.config["wavelength"]
            return sigma_s_air(wavelength=wavelength)
        else:
            return self.get_quantity("sigma_s")

    @property
    def _width(self):
        """Return scene width based on configuration."""

        # If width is not set, compute a value corresponding to an optically
        # thick layer (10x scattering mean free path)
        if self.width == "auto":
            return 10. / self._sigma_s
        else:
            return self.get_quantity("width")

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        if ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = self.phase()[f"phase_{self.id}"]

        return {
            f"medium_{self.id}": {
                "type": "homogeneous",
                "phase": phase,
                "sigma_t": {
                    "type": "uniform",
                    "value": self._sigma_s.to(kdu.get("length") ** -1).magnitude
                },
                "albedo": {
                    "type": "uniform",
                    "value": self._albedo
                },
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.media(ref=False)[f"medium_{self.id}"]

        width = self._width.to(kdu.get("length")).magnitude
        height, offset = self._height
        height = height.to(kdu.get("length")).magnitude
        offset = offset.to(kdu.get("length")).magnitude

        return {
            f"shape_{self.id}": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * width, 0., 0., 0.],
                        [0., 0.5 * width, 0., 0.],
                        [0., 0., 0.5 * (height + offset), 0.5 * (height - offset)],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
