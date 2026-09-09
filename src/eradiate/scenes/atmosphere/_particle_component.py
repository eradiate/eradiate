"""
Particle components.
"""

from __future__ import annotations

import typing as t
import warnings
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere
from ._particle_dist import ParticleDistribution, particle_distribution_factory
from ..core import traverse
from ..phase import PhaseFunction, ParticlePhaseFunction, phase_function_factory
from ...attrs import define, documented
from ...contexts import KernelContext
from ...grid import SpatialGrid
from ...kernel import SceneParameter
from ...radprops import ParticleProperties
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...units import to_quantity
from ...util.misc import cache_by_id
from ...validators import is_positive, all_positive


def _particle_distribution_converter(value):
    if isinstance(value, str):
        if value == "uniform":
            return particle_distribution_factory.convert({"type": "uniform"})
        elif value == "gaussian":
            return particle_distribution_factory.convert({"type": "gaussian"})
        elif value == "exponential":
            return particle_distribution_factory.convert({"type": "exponential"})

    return particle_distribution_factory.convert(value)


@define(eq=False, slots=False)
class ParticleComponent(AbstractHeterogeneousAtmosphere):
    """Abstract base class for particulate media"""

    @particle_properties.validator
    def _particle_properties_validator(self, attribute, value):
        if value.has_size_distribution:
            raise ValueError(
                "While initialising ParticleEnsemble: 'particle_properties' "
                "must not have 'reff'/'veff' dimensions; use ParticleField "
                "for a spatially varying composition"
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
        doc="Force the use of a polarized phase function implementation, even "
        "when no polarization information is available.",
        type="bool",
        default="False",
    )

    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        min_sigma_s = self.eval_sigma_s(ctx.si).min(axis=-1)
        # Set to inf if sigma_s is zero
        with np.errstate(divide="ignore"):
            return 1.0 / min_sigma_s


@define(eq=False, slots=False)
class ParticleAOTComponent(ParticleComponent):
    """
    Particle scene element configured by AOT [``particle_aot``].

    The particle medium has a vertical extension specified by a bottom
    altitude (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the medium, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available
    distribution types and corresponding parameters.
    The particle number fraction is evaluated on the vertical layers of the
    render grid (see ``geometry``) to describe its variations with altitude.
    The particle density in the medium is adjusted so that the particle
    particles' optical thickness at a specified reference wavelength
    (``w_ref``) meets a specified value (``tau_ref``).
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
            validator=[all_positive, pinttrs.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle component. May be a scalar (the "
        "same altitude everywhere) or an ``(n_x, n_y)`` array matching the "
        "scene geometry's horizontal grid shape, for a bottom altitude that "
        "varies spatially.\n"
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
            validator=[all_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Top altitude of the particle component. May be a scalar (the "
        "same altitude everywhere) or an ``(n_x, n_y)`` array matching the "
        "scene geometry's horizontal grid shape, for a top altitude that "
        "varies spatially.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="float or quantity",
        default="1 km",
    )

    @bottom.validator
    @top.validator
    def _bottom_top_validator(self, attribute, value):
        if np.any(self.bottom >= self.top):
            raise ValueError(
                f"while validating '{attribute.name}': bottom altitude must be "
                "lower than top altitude "
                f"(got bottom={self.bottom}, top={self.top})"
            )

    distribution: ParticleDistribution = documented(
        attrs.field(
            default="uniform",
            converter=_particle_distribution_converter,
            validator=attrs.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution in the Z dimension of the coords grid. "
        "Simple defaults can be set using a string: "
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
                "While initializing ParticleAOTComponent: the provided aerosol "
                "single-scattering property dataset does not contain the selected "
                f"reference wavelength (w_ref = {value})",
                stacklevel=2,
            )

    tau_ref: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            factory=lambda: ureg.Quantity(0.2, ureg.dimensionless),
            validator=[all_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Extinction optical thickness at the reference wavelength. May be "
        "a scalar (the same optical thickness everywhere) or an "
        "``(n_x, n_y)`` array matching the scene geometry's horizontal grid "
        "shape, for an optical thickness that varies spatially.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    @tau_ref.validator
    @bottom.validator
    @top.validator
    def _xy_shape_validator(self, attribute, value):
        if self.geometry is None:
            return
        if np.size(value) == 1:
            return
        if value.shape != self.geometry.grid.shape[:2]:
            raise ValueError(
                "While initialising ParticleAOTComponent: the shape of the "
                f"{attribute.name} attribute is inconsistent with the "
                "scene geometry. Expected a scalar value or a "
                f"{self.geometry.grid.shape[:2]} sized array, "
                f"received a {value.shape} sized array."
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

    def eval_fractions(self, grid: SpatialGrid) -> np.ndarray:
        """
        Compute the particle number fraction in the component in 3D
        according to the distribution.

        The Z dimension (vertical extent) supports spatially varying
        top and bottom in X and Y.

        Returns
        -------
        ndarray
            Particle number fractions as a (n_x, n_y, n_layers)-shaped
            array.
        """

        # Variable bottom and top altitude
        bottom = np.atleast_2d(self.bottom)
        extent = np.atleast_2d(self.top - self.bottom)
        z = (grid.layers - bottom[..., np.newaxis]) / extent[..., np.newaxis]

        fractions = self.distribution(z.m_as(ureg.dimensionless))

        out = np.zeros(fractions.shape)
        fractions_sum = fractions.sum(axis=-1, keepdims=True)
        np.divide(fractions, fractions_sum, out=out, where=fractions_sum > 0.0)

        return out

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def _eval_albedo_impl(self, w: pint.Quantity, grid: SpatialGrid) -> pint.Quantity:
        # Return albedo from dataset (without accounting for bypass switches)
        # This routine returns an array of shape (n_wavelengths, x, y, n_layers)
        pp = self.particle_properties
        interpolated = pp.eval_ssa(np.atleast_1d(w))
        fractions = self.eval_fractions(grid)
        where_present = fractions > 0
        # np.multiply.outer() does not support pint Quantity operands, so the
        # magnitude and units are combined by hand.
        albedo = np.multiply.outer(interpolated.m, where_present) * interpolated.units
        return albedo

    def _eval_sigma_t_impl(self, w: pint.Quantity, grid: SpatialGrid) -> pint.Quantity:
        # Return extinction coefficient from dataset (without accounting
        # for bypass switches). Returns an array of shape (n_wavelengths, x, y, n_layers)

        # Collect input data
        w = np.atleast_1d(w)
        pp = self.particle_properties
        sigma_t_star = pp.eval_ext(w)
        sigma_t_star_ref = pp.eval_ext(self.w_ref)

        # Compute target optical thickness value. np.multiply.outer() does
        # not support pint Quantity operands, so the magnitude and units are
        # combined by hand.
        sigma_t_star_ratio = sigma_t_star / sigma_t_star_ref
        tau_ref = np.atleast_2d(self.tau_ref)
        tau = (
            np.multiply.outer(sigma_t_star_ratio.m, tau_ref.m)
            * sigma_t_star_ratio.units
            * tau_ref.units
        )

        # Scatter this total OT to all layers
        fractions = self.eval_fractions(grid)
        tau_layers = tau[..., np.newaxis] * fractions[np.newaxis, ...]

        # Compute corresponding average coefficient
        sigma_t = tau_layers / grid.layer_height

        return sigma_t

    def _eval_sigma_a_impl(self, w: pint.Quantity, grid: SpatialGrid) -> pint.Quantity:
        # Return absorption coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_x, n_y, n_layers)
        sigma_t = self._eval_sigma_t_impl(w, grid)
        albedo = self._eval_albedo_impl(w, grid)
        return sigma_t * (1.0 - albedo.m)

    def _eval_sigma_s_impl(self, w: pint.Quantity, grid: SpatialGrid) -> pint.Quantity:
        # Return scattering coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_x, n_y, n_layers)
        sigma_t = self._eval_sigma_t_impl(w, grid)
        albedo = self._eval_albedo_impl(w, grid)
        return sigma_t * albedo.m

    @singledispatchmethod
    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_albedo.register(MonoSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_albedo_mono(si, self.geometry.grid if grid is None else grid)

    @eval_albedo.register(CKDSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_albedo_ckd(
            si=si,
            grid=self.geometry.grid if grid is None else grid,
        )

    @cache_by_id
    def eval_albedo_mono(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        w = si.w

        if self.has_absorption and self.has_scattering:
            albedo = self._eval_albedo_impl(w, grid).squeeze(axis=0)

        elif self.has_absorption and not self.has_scattering:
            albedo = 0.0 * ureg.dimensionless

        elif self.has_scattering and not self.has_absorption:
            albedo = 1.0 * ureg.dimensionless

        # Albedo is constant vs spatial dimension
        return np.full_like(grid.layers, albedo)

    def eval_albedo_ckd(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        return self.eval_albedo_mono(si, grid)

    @singledispatchmethod
    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_t.register(MonoSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_t_mono(si, self.geometry.grid if grid is None else grid)

    @eval_sigma_t.register(CKDSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_t_ckd(
            si=si,
            grid=self.geometry.grid if grid is None else grid,
        )

    @cache_by_id
    def eval_sigma_t_mono(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        w = si.w
        result = self._eval_sigma_t_impl(w, grid).squeeze(axis=0)

        if self.has_absorption and self.has_scattering:
            return result

        elif not self.has_absorption and self.has_scattering:
            return result - self._eval_sigma_a_impl(w, grid).squeeze(axis=0)

        elif self.has_absorption and not self.has_scattering:
            return result - self._eval_sigma_s_impl(w, grid).squeeze(axis=0)

    def eval_sigma_t_ckd(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        return self.eval_sigma_t_mono(si, grid)

    @singledispatchmethod
    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_a.register(MonoSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_a_mono(si, self.geometry.grid if grid is None else grid)

    @eval_sigma_a.register(CKDSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_a_ckd(
            si=si,
            grid=self.geometry.grid if grid is None else grid,
        )

    @cache_by_id
    def eval_sigma_a_mono(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        value = self._eval_sigma_a_impl(si.w, grid).squeeze(axis=0)
        return value if self.has_absorption else np.zeros_like(value) * value.units

    def eval_sigma_a_ckd(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        return self.eval_sigma_a_mono(si, grid)

    @singledispatchmethod
    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_s.register(MonoSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_s_mono(si, self.geometry.grid if grid is None else grid)

    @eval_sigma_s.register(CKDSpectralIndex)
    def _(self, si, grid: SpatialGrid | None = None) -> pint.Quantity:
        return self.eval_sigma_s_ckd(
            si=si,
            grid=self.geometry.grid if grid is None else grid,
        )

    @cache_by_id
    def eval_sigma_s_mono(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        value = self._eval_sigma_s_impl(si.w, grid).squeeze(axis=0)
        return value if self.has_scattering else np.zeros_like(value) * value.units

    def eval_sigma_s_ckd(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        return self.eval_sigma_s_mono(si, grid)

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


@define(eq=False, slots=False)
class ParticleExtinctionComponent(ParticleComponent):
    """
    Particle scene element configured by extinction and albedo
    [``particle_extinction``].

    This class is parameterized by absolute values of extinction and single
    -scattering albedo provided through spatially defined data arrays.

    ``sigma_t`` and ``albedo`` must be defined over exactly the ``(w, x, y,
    z)`` dimensions, with ``w`` wavelength and ``x``/``y``/``z`` spatial axes
    aligned with the scene geometry's grid. They are interpolated over ``w``
    and ``z``; ``x``/``y`` are used as-is, so their values must already match
    the scene geometry's horizontal grid shape.

    This class accepts any factory-convertible phase function specification.
    """

    sigma_t: xr.DataArray = documented(
        attrs.field(
            kw_only=True,
            validator=attrs.validators.instance_of(xr.DataArray),
        ),
        doc="Extinction coefficient. A ``DataArray`` whose dimensions are "
        "exactly ``(w, x, y, z)``: ``w`` is wavelength, ``x``/``y``/``z`` "
        "are spatial axes aligned with the scene geometry's grid. Values "
        "and the ``w``/``z`` coordinates must carry a ``'units'`` entry in "
        "their ``attrs``, as read by :func:`.to_quantity`. Must have "
        "matching sizes with ``albedo``.",
        type="DataArray",
        init_type="DataArray",
    )

    albedo: xr.DataArray = documented(
        attrs.field(
            kw_only=True,
            validator=attrs.validators.instance_of(xr.DataArray),
        ),
        doc="Single-scattering albedo. Same dimension and unit-metadata "
        "convention as ``sigma_t`` (dimensionless), and must have matching "
        "sizes with it. Values must lie in [0, 1].",
        type="DataArray",
        init_type="DataArray",
    )

    phase_spec: t.Any = documented(
        attrs.field(kw_only=True, converter=phase_function_factory.convert),
        doc="Phase function specification: a :class:`.PhaseFunction` "
        "instance, or a dict/string the phase function factory can "
        "convert.",
        type="Any",
        init_type="PhaseFunction or dict or str",
    )

    _phase: PhaseFunction | None = attrs.field(default=None, init=False)

    @sigma_t.validator
    @albedo.validator
    def _dims_validator(self, attribute, value):
        if set(value.dims) != {"w", "x", "y", "z"}:
            raise ValueError(
                f"while validating '{attribute.name}': dimensions must be "
                "exactly ('w', 'x', 'y', 'z')"
            )
        if self.sigma_t.sizes != self.albedo.sizes:
            raise ValueError(
                f"while validating '{attribute.name}': sigma_t and albedo must "
                f"have matching sizes (got {dict(self.sigma_t.sizes)} and "
                f"{dict(self.albedo.sizes)})"
            )

    @albedo.validator
    def _albedo_validator(self, attribute, value):
        if np.any(value.values > 1.0):
            raise ValueError("Albedo must be lower or equal to 1.0")

    def update(self) -> None:
        self._phase = self.phase_spec

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def _interp(
        self, da: xr.DataArray, si: SpectralIndex, grid: SpatialGrid
    ) -> pint.Quantity:
        def _interp_axis(da, dim, values):
            # A single-point axis is treated as uniform along that dimension
            # (e.g. a single wavelength sample applies at every wavelength).
            # Linear interpolation is undefined on a single point and
            # xarray/scipy silently return NaN there, so fall back to an
            # extrapolating nearest-neighbour lookup instead.
            if da.sizes[dim] == 1:
                return da.interp(
                    coords={dim: values},
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"},
                )
            return da.interp(coords={dim: values}, method="linear")

        result = _interp_axis(da, "w", np.atleast_1d(si.w.m_as(da.w.attrs["units"])))
        result = _interp_axis(result, "z", grid.layers.m_as(da.z.attrs["units"]))
        return to_quantity(result.squeeze("w", drop=True))

    def _eval_sigma_a_impl(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        sigma_t = self._interp(self.sigma_t, si, grid)
        albedo = self._interp(self.albedo, si, grid)
        return sigma_t * (1.0 - albedo.m)

    def _eval_sigma_s_impl(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        sigma_t = self._interp(self.sigma_t, si, grid)
        albedo = self._interp(self.albedo, si, grid)
        return sigma_t * albedo.m

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid
        result = self._interp(self.sigma_t, si, grid)

        if self.has_absorption and self.has_scattering:
            return result
        elif not self.has_absorption and self.has_scattering:
            return result - self._eval_sigma_a_impl(si, grid)
        elif self.has_absorption and not self.has_scattering:
            return result - self._eval_sigma_s_impl(si, grid)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid
        value = self._eval_sigma_a_impl(si, grid)
        return value if self.has_absorption else np.zeros_like(value) * value.units

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid
        value = self._eval_sigma_s_impl(si, grid)
        return value if self.has_scattering else np.zeros_like(value) * value.units

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid

        if self.has_absorption and self.has_scattering:
            return self._interp(self.albedo, si, grid)

        # Degenerate cases: albedo is uniformly 0 or 1, but must still match
        # sigma_t's natural (possibly 3D) shape.
        sigma_t = self._interp(self.sigma_t, si, grid)
        fill = 1.0 if self.has_scattering else 0.0
        return np.full_like(sigma_t.m, fill) * ureg.dimensionless

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
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
