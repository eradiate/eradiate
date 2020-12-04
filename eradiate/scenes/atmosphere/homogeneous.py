"""Homogeneous atmosphere scene elements."""

import attr

import eradiate
from .base import Atmosphere, _converter_or_auto, _validators_or_auto
from .radiative_properties.rayleigh import compute_sigma_s_air
from ..core import SceneElementFactory
from ...util.attrs import attrib_quantity, converter_to_units, validator_units_compatible, validator_is_positive
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

    .. rubric:: Constructor arguments / instance attributes

    ``sigma_s`` (float or "auto"):
        Atmosphere scattering coefficient value. If set to ``"auto"``,
        the scattering coefficient will be computed based on the current
        operational mode configuration using the :func:`sigma_s_air`
        function. Default: ``"auto"``.

        Unit-enabled field (default unit: cdu[collision_coefficient]).

    """

    sigma_s = attrib_quantity(
        default="auto",
        converter=_converter_or_auto(converter_to_units(cdu.generator("collision_coefficient"))),
        validator=_validators_or_auto([validator_units_compatible(ureg.m ** -1), validator_is_positive]),
        units_compatible=ureg.m ** -1,
        units_add_converter=False,
        units_add_validator=False,
    )  # TODO: turn into a Spectrum

    @property
    def kernel_height(self):
        if self.height == "auto":
            height = ureg.Quantity(100, "km")
        else:
            height = self.height

        return height.to(kdu.get("length"))

    @property
    def _albedo(self):
        """Return albedo."""
        return 1.

    @property
    def _sigma_s(self):
        """Return scattering coefficient based on configuration."""
        if self.sigma_s == "auto":
            return compute_sigma_s_air(wavelength=eradiate.mode.wavelength)
        else:
            return self.sigma_s

    @property
    def kernel_width(self):
        if self.width == "auto":
            width = 10. / self._sigma_s
        else:
            width = self.width

        return width.to(kdu.get("length"))

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        if ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = self.phase()[f"phase_{self.id}"]
        sigma_s = self._sigma_s.to(kdu.get("collision_coefficient")).magnitude
        return {
            f"medium_{self.id}": {
                "type": "homogeneous",
                "phase": phase,
                "sigma_t": {
                    "type": "uniform",
                    "value": sigma_s
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

        width = self.kernel_width.magnitude
        height = self.kernel_height.magnitude
        offset = self.kernel_offset.magnitude

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
