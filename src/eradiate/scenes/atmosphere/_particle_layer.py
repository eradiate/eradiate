"""
Particle layers.
"""

from __future__ import annotations

import warnings
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import pinttrs

from ._core import AbstractHeterogeneousAtmosphere
from ._particle_dist import ParticleDistribution, particle_distribution_factory
from ..core import traverse
from ..phase import ParticlePhaseFunction
from ...attrs import define, documented
from ...contexts import KernelContext
from ...grid import GridCoords
from ...kernel import SceneParameter
from ...radprops import ParticleProperties
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id
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


@define(eq=False, slots=False, init=False)
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

    Notes
    -----
    If the optical property dataset contains only one spectral data point, the
    data is considered uniform throughout the entire spectrum.
    """

    bottom: pint.Quantity = documented(
        pinttrs.field(
            default=ureg.Quantity(0.0, ureg.km),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="float or quantity",
        default="0 km",
    )

    top: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="float or quantity",
        default="1 km.",
    )

    @bottom.validator
    @top.validator
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
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            default=550.0 * ureg.nm,
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Reference wavelength at which the extinction optical thickness is "
        "specified. To minimize the uncertainty on the computed extinction "
        "coefficient, it is recommended that this wavelength is included in the "
        "provided radiative property dataset.\n"
        "\n"
        "Unit-enabled field (default: ucc['wavelength']).",
        type="quantity",
        init_type="quantity or float",
        default="550.0 nm",
    )

    @w_ref.validator
    def _w_ref_validator(self, attribute, value):
        w_units = self.particle_properties.w.u
        if not np.any(np.isclose(value.m_as(w_units), self.particle_properties.w.m)):
            warnings.warn(
                "While initializing ParticleLayer: the provided aerosol "
                "single-scattering property dataset does not contain the selected "
                f"reference wavelength (w_ref = {value})",
                stacklevel=2,
            )

    tau_ref: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            factory=lambda: ureg.Quantity(0.2, ureg.dimensionless),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Extinction optical thickness at the reference wavelength.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    particle_properties: ParticleProperties = documented(
        attrs.field(
            default="govaerts_2021-continental",
            converter=ParticleProperties.convert,
        ),
        doc="Particle single-scattering properties.",
        type="ParticleProperties",
        init_type="ParticleProperties or Dataset or path-like or str",
        default='"govaerts_2021-continental"',
    )

    has_absorption: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="Absorption bypass switch. If ``True``, the absorption coefficient "
        "is computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="Scattering bypass switch. If ``True``, the scattering coefficient "
        "is computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    @has_absorption.validator
    @has_scattering.validator
    def _switch_validator(self, attribute, value):
        if not self.has_absorption and not self.has_scattering:
            raise ValueError(
                f"while validating {attribute.name}: at least one of "
                "'has_absorption' and 'has_scattering' must be True"
            )

    force_polarized_phase: bool = documented(
        attrs.field(default=False, converter=bool),
        doc="Force the use of a polarized phase function implementation, even"
        "when no polarization information is available.",
        type="bool",
        default="False",
    )

    _phase: ParticlePhaseFunction | None = attrs.field(default=None, init=False)

    def __init__(self, *args, dataset=None, **kwargs):
        if dataset is not None:
            warnings.warn(
                "The 'dataset' argument is deprecated; use 'particle_properties' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("particle_properties", dataset)
        self.__attrs_init__(*args, **kwargs)

    def update(self) -> None:
        self._phase = ParticlePhaseFunction(
            id=self.phase_id,
            particle_properties=self.particle_properties,
            force_polarized=self.force_polarized_phase,
        )

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    def eval_fractions(self, grid: GridCoords) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Returns
        -------
        ndarray
            Particle number fractions as a (n_layers,)-shaped array.
        """
        x = (grid.layers - self.bottom) / (self.top - self.bottom)
        fractions = self.distribution(x.m_as(ureg.dimensionless))
        fractions /= np.sum(fractions)
        return fractions

    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        min_sigma_s = self.eval_sigma_s(ctx.si).min()
        return 1.0 / min_sigma_s if min_sigma_s != 0.0 else np.inf * ureg.m

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @cache_by_id
    def _eval_albedo_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return albedo from dataset (without accounting for bypass switches)
        # This routine is vectorized and returns an array of shape
        # (n_wavelengths, n_layers)
        pp = self.particle_properties
        albedo = pp.eval_ssa(w)
        zmask = np.reshape(self.eval_fractions(grid) > 0, (1, -1))
        return albedo * zmask

    @cache_by_id
    def _eval_sigma_t_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return extinction coefficient from dataset (without accounting
        # for bypass switches). This routine is vectorized and returns an
        # array of shape (n_wavelengths, n_layers)

        # Collect input data
        w = np.atleast_1d(w)
        pp = self.particle_properties
        sigma_t_star = pp.eval_ext(w)
        sigma_t_star_ref = pp.eval_ext(self.w_ref)

        # Compute target optical thickness value
        tau = self.tau_ref * sigma_t_star / sigma_t_star_ref

        # Scatter this total OT to all layers
        # TODO: Make sure that axis order is consistent with other vectorized
        #  routines
        fractions = self.eval_fractions(grid)
        tau_layers = np.broadcast_to(
            np.reshape(tau, (-1, 1)), (len(w), grid.n_layers)
        ) * np.reshape(fractions, (1, -1))

        # Compute corresponding average coefficient
        sigma_t = tau_layers / grid.layer_height

        return sigma_t

    def _eval_sigma_a_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return absorption coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_layers)
        albedo = self._eval_albedo_impl(w, grid)
        return self._eval_sigma_t_impl(w, grid) * (1.0 - albedo.m)

    def _eval_sigma_s_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return scattering coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_layers)
        albedo = self._eval_albedo_impl(w, grid)
        return self._eval_sigma_t_impl(w, grid) * albedo.m

    @singledispatchmethod
    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_albedo.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_albedo_mono(
            w=si.w,
            grid=self.geometry.grid if grid is None else grid,
        )

    @eval_albedo.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_albedo_ckd(
            w=si.w,
            g=si.g,
            grid=self.geometry.grid if grid is None else grid,
        )

    def eval_albedo_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        if self.has_absorption and self.has_scattering:
            albedo = self._eval_albedo_impl(w, grid).squeeze()

        elif self.has_absorption and not self.has_scattering:
            albedo = 0.0 * ureg.dimensionless

        elif self.has_scattering and not self.has_absorption:
            albedo = 1.0 * ureg.dimensionless

        else:
            raise RuntimeError

        # Albedo is constant vs spatial dimension
        return np.full_like(grid.layers, albedo)

    def eval_albedo_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_albedo_mono(w=w, grid=grid)

    @singledispatchmethod
    def eval_sigma_t(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_t.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_t_mono(
            w=si.w, grid=self.geometry.grid if grid is None else grid
        )

    @eval_sigma_t.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_t_ckd(
            w=si.w, g=si.g, grid=self.geometry.grid if grid is None else grid
        )

    def eval_sigma_t_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        result = self._eval_sigma_t_impl(w, grid).squeeze()

        if self.has_absorption and self.has_scattering:
            return result

        elif not self.has_absorption and self.has_scattering:
            return result - self._eval_sigma_a_impl(w, grid).squeeze()

        elif self.has_absorption and not self.has_scattering:
            return result - self._eval_sigma_s_impl(w, grid).squeeze()

        raise RuntimeError

    def eval_sigma_t_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_t_mono(w=w, grid=grid)

    @singledispatchmethod
    def eval_sigma_a(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_a.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_a_mono(
            w=si.w, grid=self.geometry.grid if grid is None else grid
        )

    @eval_sigma_a.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_a_ckd(
            w=si.w, g=si.g, grid=self.geometry.grid if grid is None else grid
        )

    def eval_sigma_a_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        value = self._eval_sigma_a_impl(w, grid).squeeze()
        return value if self.has_absorption else np.zeros_like(value) * value.units

    def eval_sigma_a_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_a_mono(w, grid)

    @singledispatchmethod
    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_s.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_s_mono(
            w=si.w, grid=self.geometry.grid if grid is None else grid
        )

    @eval_sigma_s.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_s_ckd(
            w=si.w, g=si.g, grid=self.geometry.grid if grid is None else grid
        )

    def eval_sigma_s_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        value = self._eval_sigma_s_impl(w, grid).squeeze()
        return value if self.has_scattering else np.zeros_like(value) * value.units

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_s_mono(w, grid)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> ParticlePhaseFunction:
        # Inherit docstring
        return self._phase

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        _, result = traverse(self.phase)
        return result.data
