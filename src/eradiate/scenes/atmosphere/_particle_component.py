"""
Particle components.
"""

from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._particle_dist import (
    ArrayParticleDistribution,
    ParticleDistribution,
    particle_distribution_factory,
)
from ..core import traverse
from ..phase import ParticlePhaseFunction, PhaseFunction, phase_function_factory
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...kernel import SceneParameter
from ...radprops import ParticleProperties
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id
from ...validators import all_positive, is_positive

if TYPE_CHECKING:
    from ._particle_psd_component import ParticlePSDComponent


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

    @classmethod
    def convert(cls, value: Any) -> ParticleComponent:
        """
        Conversion logic:

        * If value is a ParticleComponent instance, return it unchanged.
        * If value is a dictionary, it is interpreted by
          :data:`.atmosphere_factory`; the ``"type"`` parameter may be
          omitted and will automatically be set to ``"particle_aot"``.
          Other types, e.g. ``"particle_psd"``, must set ``"type"``
          explicitly.
        * Otherwise, the value is returned unchanged.
        """
        if isinstance(value, MutableMapping) and "type" not in value:
            value = {**value, "type": "particle_aot"}
        return atmosphere_factory.convert(value, allowed_cls=cls)

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

    @particle_properties.validator
    def _particle_properties_validator(self, attribute, value):
        if value.has_size_distribution:
            raise ValueError(
                "While initializing ParticleAOTComponent: 'particle_properties' "
                "must not have 'reff'/'veff' dimensions; use ParticlePSDComponent "
                "for a spatially varying composition"
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
        return self._eval_fractions_impl(
            grid, self.bottom, self.top, self.distribution
        ).copy()

    # Spectrally invariant terms are cached against the fields they depend on,
    # so that reassigning a field invalidates them.

    @cache_by_id
    def _eval_fractions_impl(
        self,
        grid: SpatialGrid,
        bottom: pint.Quantity,
        top: pint.Quantity,
        distribution: ParticleDistribution,
    ) -> np.ndarray:
        # Variable bottom and top altitude
        bottom = np.atleast_2d(bottom)
        extent = np.atleast_2d(top - bottom)
        z = (grid.layers - bottom[..., np.newaxis]) / extent[..., np.newaxis]

        fractions = distribution(z.m_as(ureg.dimensionless))

        out = np.zeros(fractions.shape)
        fractions_sum = fractions.sum(axis=-1, keepdims=True)
        np.divide(fractions, fractions_sum, out=out, where=fractions_sum > 0.0)

        return out

    @cache_by_id
    def _eval_ext_ref(
        self, particle_properties: ParticleProperties, w_ref: pint.Quantity
    ) -> pint.Quantity:
        return particle_properties.eval_ext(w_ref)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def _eval_albedo_impl(self, w: pint.Quantity, grid: SpatialGrid) -> pint.Quantity:
        # Return albedo from dataset (without accounting for bypass switches)
        # This routine returns an array of shape (n_wavelengths, x, y, n_layers)
        pp = self.particle_properties
        interpolated = pp.eval_ssa(np.atleast_1d(w))
        fractions = self._eval_fractions_impl(
            grid, self.bottom, self.top, self.distribution
        )
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
        sigma_t_star_ref = self._eval_ext_ref(pp, self.w_ref)

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
        fractions = self._eval_fractions_impl(
            grid, self.bottom, self.top, self.distribution
        )
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

    @classmethod
    def from_psd_voxel(
        cls,
        particle_psd_component: ParticlePSDComponent,
        w: pint.Quantity,
        ix: int,
        iy: int,
        iz: int,
        grid: SpatialGrid | None = None,
    ) -> ParticleAOTComponent:
        """
        Build a :class:`.ParticleAOTComponent` representing a single voxel of
        the particle PSD component at the given grid location.

        Parameters
        ----------
        particle_psd_component : :class:`.ParticlePSDComponent`
            Source particle PSD component.

        w : :class:`pint.Quantity`
            Evaluation wavelength.

        ix, iy, iz : int
            Voxel indices in the render grid.

        grid : :class:`.SpatialGrid`, optional
            Target render grid. Defaults to ``particle_psd_component.geometry.grid``.

        Returns
        -------
        :class:`.ParticleAOTComponent`

        Raises
        ------
        ValueError
            If ``(ix, iy, iz)`` is out of bounds, or the voxel there has no
            cloud (zero amount).
        """
        grid = grid or particle_psd_component.geometry.grid
        resampled_profile = particle_psd_component.profile.resample_regular(grid).data
        amount_name = particle_psd_component._amount
        binding = particle_psd_component._binding

        sizes = resampled_profile.sizes
        in_bounds = (
            0 <= ix < sizes["x"] and 0 <= iy < sizes["y"] and 0 <= iz < sizes["z"]
        )

        has_cloud = (
            in_bounds and float(resampled_profile[amount_name].values[iz, iy, ix]) > 0.0
        )
        if not has_cloud:
            raise ValueError(f"No cloud voxel found at ({ix}, {iy}, {iz}).")

        profile_point = resampled_profile.isel(x=ix, y=iy, z=iz)

        col_units = uck.get("collision_coefficient")
        wavelength = np.atleast_1d(w)
        pp = particle_psd_component.properties

        reff_qty = to_quantity(profile_point[binding["reff"]]).item()
        veff_qty = to_quantity(profile_point[binding["veff"]]).item()
        amount_qty = to_quantity(profile_point[amount_name]).item()

        i_reff = int(
            np.argmin(np.abs(pp.reff.m_as(ureg.micron) - reff_qty.m_as(ureg.micron)))
        )
        i_veff = int(
            np.argmin(
                np.abs(
                    pp.veff.m_as(ureg.dimensionless) - veff_qty.m_as(ureg.dimensionless)
                )
            )
        )

        # Evaluate on the wavelength axis first, then pick this exact point.
        m_extinction = pp.eval_ext(wavelength)[0, i_reff, i_veff]
        sigma_t = (m_extinction * amount_qty).to(col_units)

        layer_height = to_quantity(
            xr.DataArray(np.diff(grid.levels.m_as(ureg.km)), attrs={"units": "km"})
        )[iz]

        tgt_edges_x = grid.edges_x.m_as(ureg.km)
        tgt_edges_y = grid.edges_y.m_as(ureg.km)
        src_x0, src_x1 = float(tgt_edges_x[ix]), float(tgt_edges_x[ix + 1])
        src_y0, src_y1 = float(tgt_edges_y[iy]), float(tgt_edges_y[iy + 1])

        def _compute_overlap(
            src_x0: float,
            src_x1: float,
            src_y0: float,
            src_y1: float,
            tgt_edges_x: np.ndarray,
            tgt_edges_y: np.ndarray,
        ) -> np.ndarray:
            """Return the ``(n_x, n_y)`` fractional area overlap between the
            source cell and each target grid cell."""
            overlap_x = np.maximum(
                0.0,
                np.minimum(src_x1, tgt_edges_x[1:])
                - np.maximum(src_x0, tgt_edges_x[:-1]),
            ) / (tgt_edges_x[1:] - tgt_edges_x[:-1])
            overlap_y = np.maximum(
                0.0,
                np.minimum(src_y1, tgt_edges_y[1:])
                - np.maximum(src_y0, tgt_edges_y[:-1]),
            ) / (tgt_edges_y[1:] - tgt_edges_y[:-1])
            return np.outer(overlap_x, overlap_y)

        overlap = _compute_overlap(
            src_x0, src_x1, src_y0, src_y1, tgt_edges_x, tgt_edges_y
        )
        tau_ref = (sigma_t * layer_height * overlap).to(ureg.dimensionless)

        phase_ds = pp.phase_dataset_for_reff_veff(
            wavelength[0], i_reff, i_veff, sigma_t
        )

        geometry = (
            particle_psd_component.geometry
            if grid is particle_psd_component.geometry.grid
            else attrs.evolve(particle_psd_component.geometry, grid=grid)
        )

        return cls(
            bottom=grid.levels[iz],
            top=grid.levels[iz + 1],
            distribution="uniform",
            w_ref=wavelength[0],
            tau_ref=tau_ref,
            particle_properties=phase_ds,
            geometry=geometry,
            has_absorption=particle_psd_component.has_absorption,
            has_scattering=particle_psd_component.has_scattering,
        )

    @classmethod
    def from_psd_composition(
        cls,
        particle_psd_component: ParticlePSDComponent,
        w: pint.Quantity,
        i_reff: int,
        i_veff: int,
        grid: SpatialGrid | None = None,
        ext: pint.Quantity | None = None,
    ) -> ParticleAOTComponent:
        """
        Build a single :class:`.ParticleAOTComponent` approximating the whole
        component with one fixed ``(reff, veff)`` grid point.

        Every voxel's own ``reff``/``veff`` is ignored in favour of the
        given grid point; only its amount is used to build
        the per-voxel density (``tau_ref`` and ``distribution``).

        Parameters
        ----------
        particle_psd_component : :class:`.ParticlePSDComponent`
            Source particle PSD component.

        w : :class:`pint.Quantity`
            Evaluation wavelength.

        i_reff, i_veff : int
            Index along ``particle_psd_component.properties``'s ``reff``/``veff``
            dimensions, assumed constant across the whole component.

        grid : :class:`.SpatialGrid`, optional
            Target render grid. Defaults to ``particle_psd_component.geometry.grid``.

        ext : :class:`pint.Quantity`, optional
            Extinction coefficient stored in the output ParticleAOTComponent's
            ``particle_properties``. Defaults to the native grid value at
            ``(i_reff, i_veff)``. Since that dataset only ever has one
            wavelength point, this value is only ever compared to itself
            (``ext(w) / ext(w_ref)``) and has no effect on the output.

        Returns
        -------
        :class:`.ParticleAOTComponent`
        """
        grid = grid or particle_psd_component.geometry.grid
        col_units = uck.get("collision_coefficient")
        length_units = uck.get("length")
        wavelength = np.atleast_1d(w)
        pp = particle_psd_component.properties

        # Evaluate on the wavelength axis first, then pick this exact point.
        m_extinction = pp.eval_ext(wavelength)[0, i_reff, i_veff]

        resampled = particle_psd_component.profile.resample_regular(grid)
        amount = resampled[particle_psd_component._amount]
        sigma_t = m_extinction * amount  # (z, y, x)

        sigma_t_grid = np.transpose(sigma_t.m_as(col_units), (2, 1, 0))  # ->(x,y,z)

        layer_height = np.diff(grid.levels.m_as(length_units))
        tau_per_voxel = sigma_t_grid * layer_height[np.newaxis, np.newaxis, :]
        tau_ref = ureg.Quantity(tau_per_voxel.sum(axis=-1), ureg.dimensionless)

        z_coords = (grid.layers - grid.levels[0]) / (grid.levels[-1] - grid.levels[0])
        z_coords = np.broadcast_to(
            z_coords.m_as(ureg.dimensionless), tau_per_voxel.shape
        )
        distribution = ArrayParticleDistribution(
            values=tau_per_voxel, coords=z_coords, method="nearest"
        )

        phase_ds = pp.phase_dataset_for_reff_veff(wavelength[0], i_reff, i_veff, ext)

        geometry = (
            particle_psd_component.geometry
            if grid is particle_psd_component.geometry.grid
            else attrs.evolve(particle_psd_component.geometry, grid=grid)
        )

        return cls(
            bottom=np.broadcast_to(grid.levels[0], (grid.n_cells_x, grid.n_cells_y)),
            top=np.broadcast_to(grid.levels[-1], (grid.n_cells_x, grid.n_cells_y)),
            distribution=distribution,
            w_ref=wavelength[0],
            tau_ref=tau_ref,
            particle_properties=phase_ds,
            geometry=geometry,
            has_absorption=particle_psd_component.has_absorption,
            has_scattering=particle_psd_component.has_scattering,
        )


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

    phase_spec: Any = documented(
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
