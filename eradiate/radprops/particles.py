"""
Particle layers.
"""
from __future__ import annotations

from typing import Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

from .particle_dist import (
    ParticleDistribution,
    UniformParticleDistribution,
    particle_distribution_factory,
)
from .. import path_resolver
from ..attrs import documented, parse_docs
from ..contexts import SpectralContext
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import is_positive


@parse_docs
@attr.s
class ParticleLayer:
    """
    Particle layer.

    The particle layer has a vertical extension specified by a bottom altitude
    (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the layer, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :class:`.ParticleDistributionFactory` for the available distribution
    types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow to describe the variations of the particles number
    with altitude.
    The total number of particles in the layer is adjusted so that the
    particle layer's optical thickness at 550 nm meet a specified value
    (``tau_550``).
    The particles radiative properties are specified by a data set
    (``dataset``).
    """

    bottom = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="float",
        default="0 km",
    )

    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="1 km.",
    )

    @bottom.validator
    @top.validator
    def _bottom_and_top_validator(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    distribution = documented(
        attr.ib(
            factory=UniformParticleDistribution,
            converter=particle_distribution_factory.convert,
            validator=attr.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution.",
        type="dict or :class:`ParticleDistribution`",
        default=":class:`Uniform`",
    )

    tau_550 = documented(
        pinttr.ib(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the wavelength of 550 nm.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="float",
        default="0.2",
    )

    n_layers = documented(
        attr.ib(
            default=16,
            converter=int,
            validator=attr.validators.instance_of(int),
        ),
        doc="Number of layers inside the particle layer.\n",
        type="int",
        default="16",
    )

    dataset = documented(
        attr.ib(
            default=path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc"),
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Particles radiative properties data set path.",
        type="str",
    )

    @property
    def z_level(self) -> pint.Quantity:
        """
        Compute the level altitude mesh within the particle layer.

        The level altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer + 1``.

        Returns → :class:`~pint.Quantity`:
            Level altitude mesh.
        """
        return np.linspace(start=self.bottom, stop=self.top, num=self.n_layers + 1)

    @property
    def z_layer(self) -> pint.Quantity:
        """
        Compute the layer altitude mesh within the particle layer.

        The layer altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer``.

        Returns → :class:`~pint.Quantity`:
            Layer altitude mesh.
        """
        z_level = self.z_level
        return (z_level[:-1] + z_level[1:]) / 2.0

    def eval_fractions(self, z_layer: ureg.Quantity = None) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Parameter ``z_layer`` (:class:`~pint.Quantity`):
            Layer altitude mesh onto which the fractions must be computed.

        Returns → :class:`~numpy.ndarray`:
            Particles fractions.
        """
        z_layer = self.z_layer if z_layer is None else z_layer
        return self.distribution.eval_fraction(z_layer)

    def eval_phase(self, spectral_ctx: SpectralContext) -> xr.DataArray:
        """
        Evaluate the phase function.

        The phase function is represented by a :class:`~xarray.DataArray` with
        a :math:`\\mu` (``mu``) coordinate for the scattering angle cosine
        (:math:`\\mu \\in [-1, 1]`).

        Returns → :class:`~xarray.DataArray`:
            Phase function.
        """
        ds = xr.open_dataset(self.dataset)
        return (
            ds.phase.sel(i=0)
            .sel(j=0)
            .interp(w=spectral_ctx.wavelength.magnitude, kwargs=dict(bounds_error=True))
        )

    def eval_albedo(
        self, spectral_ctx: SpectralContext, z_level: ureg.Quantity = None
    ) -> pint.Quantity:
        """
        Evaluate albedo given a spectral context.

        Parameter ``z_level`` (:class:`~pint.Quantity`):
            Level altitude mesh onto which the fractions must be computed.

        Returns → :class:`~pint.Quantity`:
            Particle layer albedo.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_albedo = ds.albedo.interp(w=wavelength)
        albedo = to_quantity(interpolated_albedo)
        n_layers = self.n_layers if z_level is None else len(z_level) - 1
        albedo_array = albedo * np.ones(n_layers)
        return albedo_array

    def eval_sigma_t(
        self, spectral_ctx: SpectralContext, z_level: ureg.Quantity = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameter ``z_level`` (:class:`~pint.Quantity`):
            Level altitude mesh onto which the fractions must be computed.

        Returns → :class:`~pint.Quantity`:
            Particle layer extinction coefficient.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_sigma_t = ds.sigma_t.interp(w=wavelength)
        sigma_t = to_quantity(interpolated_sigma_t)
        z_layer = None if z_level is None else (z_level[1:] + z_level[:-1]) / 2.0
        fractions = self.eval_fractions(z_layer=z_layer)
        sigma_t_array = sigma_t * fractions
        dz = (
            (self.top - self.bottom) / self.n_layers
            if z_level is None
            else z_level[1:] - z_level[:-1]
        )
        normalised_sigma_t_array = self._normalise_to_tau(
            ki=sigma_t_array.magnitude,
            dz=dz,
            tau=self.tau_550,
        )
        return normalised_sigma_t_array

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Returns → :class:`~pint.Quantity`:
            Particle layer absorption coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) - self.eval_sigma_a(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Returns → :class:`~pint.Quantity`:
            Particle layer scattering coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) * self.eval_albedo(spectral_ctx)

    @classmethod
    def convert(cls, value: Union[ParticleLayer, dict]):
        """
        Object converter method.

        If ``value`` is a dictionary, this method forwards it to
        :meth:`from_dict`. Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.from_dict(value)

        return value

    def radprops(
        self, spectral_ctx: SpectralContext, z_level: ureg.Quantity = None
    ) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties profile of the
        particle layer.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Parameter ``z_level`` (:class:`~pint.Quantity`):
            Level altitude mesh.
            This parameter allows to compute the radiative properties of the
            particle layer on a level altitude mesh that is different from
            the "native" (specified by ``n_layers``) particle layer mesh.
            If ``None``, the level altitude mesh is the regular mesh defined
            by ``bottom``, ``top`` and ``n_layers`` for the start value,
            stop value and number of points, respectively.

        Returns → :class:`~xarray.Dataset`:
            Particle layer radiative properties profile dataset.
        """
        z_level = self.z_level if z_level is None else z_level
        z_layer = (z_level[1:] + z_level[:-1]) / 2.0
        sigma_t = self.eval_sigma_t(spectral_ctx=spectral_ctx, z_level=z_level)
        albedo = self.eval_albedo(spectral_ctx=spectral_ctx, z_level=z_level)
        wavelength = spectral_ctx.wavelength
        return xr.Dataset(
            data_vars={
                "sigma_t": (
                    ("w", "z_layer"),
                    np.atleast_2d(sigma_t.magnitude),
                    dict(
                        standard_name="extinction_coefficient",
                        long_name="extinction coefficient",
                        units=sigma_t.units,
                    ),
                ),
                "albedo": (
                    ("w", "z_layer"),
                    np.atleast_2d(albedo.magnitude),
                    dict(
                        standard_name="albedo",
                        long_name="albedo",
                        units=albedo.units,
                    ),
                ),
            },
            coords={
                "z_layer": (
                    "z_layer",
                    z_layer.magnitude,
                    dict(
                        standard_name="layer_altitude",
                        long_name="layer altitude",
                        units=z_layer.units,
                    ),
                ),
                "z_level": (
                    "z_level",
                    z_level.magnitude,
                    dict(
                        standard_name="level_altitude",
                        long_name="level altitude",
                        units=z_level.units,
                    ),
                ),
                "w": (
                    "w",
                    [
                        round(wavelength.magnitude, 12)
                    ],  # fix floating point arithmetic issue
                    dict(
                        standard_name="wavelength",
                        long_name="wavelength",
                        units=wavelength.units,
                    ),
                ),
            },
        )

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalise_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> np.ndarray:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::
            \sum_i k_i \Delta z = \tau_{550}
        where :math:`tau` is the particle layer optical thickness.

        Parameter ``ki`` (:class:`~pint.Quantity` or :class:`~np.ndarray`):
            Dimensionless extinction coefficients values [].

        Parameter ``dz`` (:class:`~pint.Quantity` or :class:`~np.ndarray`):
            Layer divisions thickness [km].

        Parameter ``tau`` (float):
            Layer optical thickness (dimensionless).

        Returns → :class:`~pint.Quantity`
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)
