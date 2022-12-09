"""
Particle layers.
"""
from __future__ import annotations

import typing as t
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere
from ._particle_dist import ParticleDistribution, particle_distribution_factory
from ..core import KernelDict
from ..phase import TabulatedPhaseFunction
from ... import converters, data
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...spectral_index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ...units import symbol, to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


def _particle_layer_distribution_converter(value):
    if isinstance(value, str):
        if value == "uniform":
            return particle_distribution_factory.convert({"type": "uniform"})
        elif value == "gaussian":
            return particle_distribution_factory.convert({"type": "gaussian"})
        elif value == "exponential":
            return particle_distribution_factory.convert({"type": "exponential"})

    return particle_distribution_factory.convert(value)


@parse_docs
@attrs.define
class ParticleLayer(AbstractHeterogeneousAtmosphere):
    """
    Particle layer scene element [``particle_layer``].

    The particle layer has a vertical extension specified by a bottom altitude
    (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the layer, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available
    distribution types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow to describe the variations of the particles number
    with altitude.
    The particle density in the layer is adjusted so that the particle layer's
    optical thickness at a specified reference wavelength (``w_ref``) meets a
    specified value (``tau_ref``).
    The particles radiative properties are specified by a data set
    (``dataset``).
    """

    _bottom: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.km),
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
        pinttr.field(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
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
    def _bottom_top_validator(self, attribute, value):
        if self.bottom >= self.top:
            raise ValueError(
                f"while validating '{attribute.name}': bottom altitude must be "
                "lower than top altitude "
                f"(got bottom={self.bottom}, top={self.top})"
            )

    distribution: ParticleDistribution = documented(
        attrs.field(
            default="uniform",
            converter=_particle_layer_distribution_converter,
            validator=attrs.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution. Simple defaults can be set using a string: "
        '``"uniform"`` (resp. ``"gaussian"``, ``"exponential"``) is converted to '
        ":class:`UniformParticleDistribution() <.UniformParticleDistribution>` "
        "(resp. :class:`GaussianParticleDistribution() <.GaussianParticleDistribution>`, "
        ":class:`ExponentialParticleDistribution() <.ExponentialParticleDistribution>`).",
        init_type=":class:`.ParticleDistribution` or dict or "
        '{"uniform", "gaussian", "exponential"}, optional',
        type=":class:`.ParticleDistribution`",
        default='"uniform"',
    )

    w_ref: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            default=550 * ureg.nm,
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Reference wavelength at which the extinction optical thickness is "
        "specified.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="550.0",
    )

    tau_ref: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the reference wavelength.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    n_layers: int = documented(
        attrs.field(
            default=16,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Number of layers in which the particle layer is discretised.",
        type="int",
        default="16",
    )

    dataset: xr.Dataset = documented(
        attrs.field(
            default="govaerts_2021-continental",
            converter=converters.to_dataset(
                load_from_id=lambda x: data.load_dataset(
                    f"spectra/particles/{x}.nc",
                )
            ),
            validator=attrs.validators.instance_of(xr.Dataset),
        ),
        doc="Particle radiative property data set."
        "If a xarray.Dataset is passed, the dataset is used as is "
        "(refer to the data guide for the format requirements of this dataset)."
        "If a path is passed, the converter tries to open the corresponding "
        "file on the hard drive; should that fail, it queries the Eradiate data"
        "store with that path."
        "If a string is passed, it is interpreted as an identifier for a "
        "particle radiative property dataset in the Eradiate data store.",
        type="Dataset",
        init_type="Dataset or path-like or str, optional",
        default="govaerts_2021-continental",
    )

    _phase: t.Optional[TabulatedPhaseFunction] = attrs.field(default=None, init=False)


    @property
    def is_molecular(self) -> bool:
        return False

    def update(self) -> None:
        super().update()

        ds = self.dataset
        self._phase = TabulatedPhaseFunction(id=self.id_phase, data=ds.phase)

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def top(self) -> pint.Quantity:
        return self._top

    @property
    def bottom(self) -> pint.Quantity:
        return self._bottom

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
        return 0.5 * (z_level[:-1] + z_level[1:])

    def eval_fractions(self) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Returns
        -------
        ndarray
            Particle fractions.
        """
        x = (self.z_layer - self.bottom) / (self.top - self.bottom)
        fractions = self.distribution(x.magnitude)
        fractions /= np.sum(fractions)
        return fractions

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        min_sigma_s = self.eval_sigma_s(spectral_index=ctx.spectral_index).min()
        return 1.0 / min_sigma_s if min_sigma_s != 0.0 else np.inf * ureg.m

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @singledispatchmethod
    def eval_albedo(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate albedo at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_albedo.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_albedo_mono(spectral_index.w)

    @eval_albedo.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_albedo_ckd(spectral_index.w, spectral_index.g)

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        ds = self.dataset
        interpolated_albedo = ds.albedo.interp(w=w.m_as(ds.w.attrs["units"]))
        return to_quantity(interpolated_albedo) * np.ones(self.n_layers)

    def eval_albedo_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_albedo_mono(w)

    @singledispatchmethod
    def eval_sigma_t(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """
        raise NotImplementedError
    
    @eval_sigma_t.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_t_mono(spectral_index.w)
    
    @eval_sigma_t.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_t_ckd(spectral_index.w, spectral_index.g)

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        ds = self.dataset
        ds_w_units = ureg(ds.w.attrs["units"])
        wavelength = w.m_as(ds_w_units)
        xs_t = to_quantity(ds.sigma_t.interp(w=wavelength))
        xs_t_ref = to_quantity(ds.sigma_t.interp(w=self.w_ref.m_as(ds_w_units)))
        fractions = self.eval_fractions()
        sigma_t_array = xs_t_ref * fractions
        dz = (self.top - self.bottom) / self.n_layers
        normalized_sigma_t_array = self._normalize_to_tau(
            ki=sigma_t_array.magnitude,
            dz=dz,
            tau=self.tau_ref,
        )
        return normalized_sigma_t_array * xs_t / xs_t_ref

    def eval_sigma_t_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_sigma_t_mono(w)


    @singledispatchmethod
    def eval_sigma_a(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Particle layer extinction coefficient.
        """
        raise NotImplementedError
    
    @eval_sigma_a.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_a_mono(spectral_index.w)
    
    @eval_sigma_a.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_a_ckd(spectral_index.w, spectral_index.g)

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_t_mono(w) - self.eval_sigma_s_mono(w)

    def eval_sigma_a_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_sigma_t_ckd(w, g) - self.eval_sigma_s_ckd(w, g)

    @singledispatchmethod
    def eval_sigma_s(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Particle layer scattering coefficient.
        """
        raise NotImplementedError
    
    @eval_sigma_s.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_s_mono(spectral_index.w)
    
    @eval_sigma_s.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_s_ckd(spectral_index.w, spectral_index.g)

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_t_mono(w) * self.eval_albedo_mono(w)

    def eval_sigma_s_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_sigma_t_ckd(w=w, g=g) * self.eval_albedo_ckd(w=w, g=g)

    @singledispatchmethod
    def eval_radprops(
        self, spectral_index: SpectralIndex, optional_fields: bool = False
    ) -> xr.Dataset:
        """
        Evaluate particle layer radiative properties at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        optional_fields : bool, optional, default: False
            If ``True``, extra the optional ``sigma_a`` and ``sigma_s`` fields,
            not required for scene construction but useful for analysis and
            debugging.

        Returns
        -------
        Dataset
            Particle layer radiative properties profile dataset.
        """
        raise NotImplementedError
    
    @eval_radprops.register
    def _(self, spectral_index: MonoSpectralIndex) -> xr.Dataset:
        return self.eval_radprops_mono(spectral_index.w)
        
    @eval_radprops.register
    def _(self, spectral_index: CKDSpectralIndex) -> xr.Dataset:
        return self.eval_radprops_ckd(spectral_index.w, spectral_index.g)
    
    def eval_radprops_mono(
        self,
        w: pint.Quantity,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        sigma_t = self.eval_sigma_t_mono(w=w)
        albedo = self.eval_albedo_mono(w=w)
        data_vars = {
            "sigma_t": (
                "z_layer",
                np.atleast_1d(sigma_t.magnitude),
                dict(
                    standard_name="volume_extinction_coefficient",
                    long_name="extinction coefficient",
                    units=symbol(sigma_t.units),
                ),
            ),
            "albedo": (
                "z_layer",
                np.atleast_1d(albedo.magnitude),
                dict(
                    standard_name="single_scattering_albedo",
                    long_name="albedo",
                    units=symbol(albedo.units),
                ),
            ),
        }

        if optional_fields:
            sigma_a = self.eval_sigma_a(spectral_index=spectral_index)
            sigma_s = self.eval_sigma_s(spectral_index=spectral_index)

            data_vars.update(
                {
                    "sigma_a": (
                        "z_layer",
                        np.atleast_1d(sigma_a.magnitude),
                        dict(
                            standard_name="volume_absorption_coefficient",
                            long_name="absorption coefficient",
                            units=symbol(sigma_a.units),
                        ),
                    ),
                    "sigma_s": (
                        "z_layer",
                        np.atleast_1d(sigma_s.magnitude),
                        dict(
                            standard_name="volume_scattering_coefficient",
                            long_name="scattering coefficient",
                            units=symbol(sigma_s.units),
                        ),
                    ),
                }
            )

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "z_layer": (
                    "z_layer",
                    self.z_layer.magnitude,
                    dict(
                        standard_name="layer_altitude",
                        long_name="layer altitude",
                        units=symbol(self.z_layer.units),
                    ),
                ),
                "z_level": (
                    "z_level",
                    self.z_level.magnitude,
                    dict(
                        standard_name="level_altitude",
                        long_name="level altitude",
                        units=symbol(self.z_level.units),
                    ),
                ),
                "w": (
                    "w",
                    [w.magnitude],
                    dict(
                        standard_name="radiation_wavelength",
                        long_name="wavelength",
                        units=symbol(w.units),
                    ),
                ),
            },
        ).isel(w=0)

    def eval_radprops_ckd(
        self,
        w: pint.Quantity,
        g: float,
        optional_fields: bool = False,
    ) -> xr.Dataset:
        return self.eval_radprops_mono(w=w, optional_fields=optional_fields)

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalize_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> pint.Quantity:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::
           \sum_i k_i \Delta z = \tau

        where :math:`\tau` is the particle layer optical thickness.

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

    @property
    def phase(self) -> TabulatedPhaseFunction:
        return self._phase

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        return self.phase.kernel_dict(ctx)
