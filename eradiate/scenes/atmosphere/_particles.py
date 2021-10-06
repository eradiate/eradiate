"""
Particle layers.
"""
from __future__ import annotations

import pathlib
import typing as t
import warnings

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._particle_dist import (
    ParticleDistribution,
    UniformParticleDistribution,
    particle_distribution_factory,
)
from ..core import KernelDict
from ... import path_resolver
from ..._mode import ModeFlags
from ...attrs import AUTO, documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import OverriddenValueWarning, UnsupportedModeError
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


@atmosphere_factory.register(type_id="particle_layer")
@parse_docs
@attr.s
class ParticleLayer(AbstractHeterogeneousAtmosphere):
    """
    Particle layer scene element [``particle_layer``].

    The particle layer has a vertical extension specified by a bottom altitude
    (set by `bottom`) and a top altitude (set by `top`).
    Inside the layer, the particles number is distributed according to a
    distribution (set by `distribution`).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available
    distribution types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (`n_layers`) to allow to describe the variations of the particles number
    with altitude.
    The total number of particles in the layer is adjusted so that the
    particle layer's optical thickness at 550 nm meet a specified value
    (`tau_550`).
    The particles radiative properties are specified by a data set
    (`dataset`).
    """

    _bottom: pint.Quantity = documented(
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
        type="quantity",
        init_type="float or quantity",
        default="0 km",
    )

    _top: pint.Quantity = documented(
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
        type="quantity",
        init_type="float or quantity",
        default="1 km.",
    )

    @_bottom.validator
    @_top.validator
    def _bottom_and_top_validator(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    distribution: ParticleDistribution = documented(
        attr.ib(
            factory=UniformParticleDistribution,
            converter=particle_distribution_factory.convert,
            validator=attr.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution.",
        init_type=":class:`ParticleDistribution` or dict",
        type=":class:`ParticleDistribution`",
        default=":class:`Uniform`",
    )

    tau_550: pint.Quantity = documented(
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
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    n_layers: int = documented(
        attr.ib(
            default=16,
            converter=int,
            validator=attr.validators.instance_of(int),
        ),
        doc="Number of layers inside the particle layer.",
        type="int",
        default="16",
    )

    dataset: str = documented(
        attr.ib(
            default=path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc"),
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Particles radiative properties data set path.",
        type="str",
    )

    weight_filename: t.Optional[str] = documented(
        attr.ib(
            default="weight.vol",
            converter=attr.converters.optional(str),
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Name of the weight volume data file if the phase function is of "
        'type ``"blendphase"``.',
        type="str",
        default='"weight.vol"',
    )

    # --------------------------------------------------------------------------
    #                        Volume data files
    # --------------------------------------------------------------------------

    @property
    def weight_file(self) -> pathlib.Path:
        return self.cache_dir / self.weight_filename

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def top(self) -> pint.Quantity:
        return self._top

    @property
    def bottom(self) -> pint.Quantity:
        return self._bottom

    def eval_width(self, ctx: t.Optional[KernelDictContext]) -> pint.Quantity:
        if ctx is not None and ctx.override_scene_width is not None:
            if self.width is not AUTO:
                warnings.warn(OverriddenValueWarning("Overriding particle layer width"))
            return ctx.override_scene_width

        else:
            if self.width is AUTO:
                spectral_ctx = ctx.spectral_ctx if ctx is not None else None
                return 10.0 / self.eval_sigma_s(spectral_ctx=spectral_ctx).min()
            else:
                return self.width

    @property
    def z_level(self) -> pint.Quantity:
        """
        Compute the level altitude mesh within the particle layer.

        The level altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer + 1``.

        Returns
        -------
        quantity
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

        Returns
        -------
        quantity
            Layer altitude mesh.
        """
        z_level = self.z_level
        return (z_level[:-1] + z_level[1:]) / 2.0

    def eval_fractions(self) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Returns
        -------
        ndarray
            Particles fractions.
        """
        return self.distribution.eval_fraction(self.z_layer)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_phase(self, spectral_ctx: SpectralContext) -> xr.DataArray:
        """
        Evaluate the phase function.

        The phase function is represented by a :class:`~xarray.DataArray` with
        a :math:`\\mu` (``mu``) coordinate for the scattering angle cosine
        (:math:`\\mu \\in [-1, 1]`).

        Returns
        -------
        DataArray
            Phase function.
        """
        ds = xr.open_dataset(self.dataset)
        if eradiate.mode().has_flags(ModeFlags.ANY_RGB):
            w = (550.0 * ureg.nm).m_as(ds.w.units)
        else:
            w = spectral_ctx.wavelength.m_as(ds.w.units)

        return ds.phase.sel(i=0, j=0).interp(
            w=w,
            mu=np.linspace(-1, 1, 201),
            kwargs=dict(bounds_error=True),
        )

    def eval_albedo(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialised methods depending on the active
        mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_albedo_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_albedo_ckd(spectral_ctx.bindex).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_RGB):
            return self.eval_albedo_mono(550.0 * ureg.nm).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        ds = xr.open_dataset(self.dataset)
        wavelengths = w.m_as(ds.w.attrs["units"])
        interpolated_albedo = ds.albedo.interp(w=wavelengths)
        albedo = to_quantity(interpolated_albedo)
        albedo_array = albedo * np.ones(self.n_layers)
        return albedo_array

    def eval_albedo_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_t(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_t_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_t_ckd(spectral_ctx.bindex).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_RGB):
            return self.eval_sigma_t_mono(550.0 * ureg.nm).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:

        ds = xr.open_dataset(self.dataset)

        # find the extinction data variable
        for dv in ds.data_vars:
            standard_name = ds[dv].standard_name
            if "extinction" in standard_name:
                extinction = ds[dv]

        ds_w_units = ureg.Unit(ds.w.attrs["units"])
        wavelength = w.m_as(ds_w_units)
        xs_t = to_quantity(extinction.interp(w=wavelength))
        xs_t_550 = to_quantity(
            extinction.interp(w=ureg.convert(550.0, ureg.nm, ds_w_units))
        )
        fractions = self.eval_fractions()
        sigma_t_array = xs_t_550 * fractions
        dz = (self.top - self.bottom) / self.n_layers
        normalized_sigma_t_array = self._normalize_to_tau(
            ki=sigma_t_array.magnitude,
            dz=dz,
            tau=self.tau_550,
        )
        return normalized_sigma_t_array * xs_t / xs_t_550

    def eval_sigma_t_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """
        # TODO: no z_level here?
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_a_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_a_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Particle layer absorption coefficient.
        """
        return self.eval_sigma_t_mono(w) - self.eval_sigma_s_mono(w)

    def eval_sigma_a_ckd(self, *bindexes: Bindex):
        raise NotImplementedError

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        quantity
            Particle layer scattering coefficient.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_a_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_a_ckd(spectral_ctx.bindex).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_RGB):
            return self.eval_sigma_a_mono(550.0 * ureg.nm).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_t_mono(w) * self.eval_albedo_mono(w)

    def eval_sigma_s_ckd(self, *bindexes: Bindex):
        raise NotImplementedError

    def eval_radprops(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties profile of the
        particle layer.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        Dataset
            Particle layer radiative properties profile dataset.
        """
        # TODO: Rename eval_dataset()?

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_RGB):
            sigma_t = self.eval_sigma_t(spectral_ctx=spectral_ctx)
            albedo = self.eval_albedo(spectral_ctx=spectral_ctx)
            wavelength = (
                spectral_ctx.wavelength
                if eradiate.mode().has_flags(ModeFlags.ANY_MONO)
                else 550.0 * ureg.nm
            )
            return xr.Dataset(
                data_vars={
                    "sigma_t": (
                        "z_layer",
                        np.atleast_1d(sigma_t.magnitude),
                        dict(
                            standard_name="extinction_coefficient",
                            long_name="extinction coefficient",
                            units=sigma_t.units,
                        ),
                    ),
                    "albedo": (
                        "z_layer",
                        np.atleast_1d(albedo.magnitude),
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
                        self.z_layer.magnitude,
                        dict(
                            standard_name="layer_altitude",
                            long_name="layer altitude",
                            units=self.z_layer.units,
                        ),
                    ),
                    "z_level": (
                        "z_level",
                        self.z_level.magnitude,
                        dict(
                            standard_name="level_altitude",
                            long_name="level altitude",
                            units=self.z_level.units,
                        ),
                    ),
                    "w": (
                        "w",
                        [wavelength.magnitude],
                        dict(
                            standard_name="wavelength",
                            long_name="wavelength",
                            units=wavelength.units,
                        ),
                    ),
                },
            ).isel(w=0)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "rgb"))

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalize_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> pint.Quantity:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::
            \sum_i k_i \Delta z = \tau_{550}

        where :math:`tau` is the particle layer optical thickness.

        Parameters
        ----------
        ki : quantity or ndarray
            Dimensionless extinction coefficients values [].

        dz : quantity or ndarray
            Layer divisions thickness [km].

        tau : float
            Layer optical thickness (dimensionless).

        Returns
        -------
        quantity
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        particles_phase = self.eval_phase(spectral_ctx=ctx.spectral_ctx)
        return KernelDict(
            {
                f"phase_{self.id}": {
                    "type": "lutphase",
                    "values": ",".join([str(value) for value in particles_phase.data]),
                }
            }
        )
