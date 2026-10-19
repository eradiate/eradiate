"""
Particle PSD components.

A particle PSD component is an atmospheric medium whose optical properties
vary spatially, described by a dense volumetric :class:`.ParticleProfile`
(per-voxel microphysical properties) and a Prt v1 particle properties
lookup dataset (wrapped in a :class:`.ParticleProperties`).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import attrs
import numpy as np
import pint
import xarray as xr

from ._core import atmosphere_factory
from ._particle_component import ParticleAOTComponent, ParticleComponent
from ._particle_profile import ParticleProfile
from ..core import traverse
from ..geometry import PlaneParallelGeometry
from ..phase import GriddedParticlePhaseFunction
from ...attrs import define, documented, get_doc
from ...contexts import KernelContext
from ...data.convert import make_prt_psd_v1
from ...grid import CartesianGrid, SpatialGrid
from ...kernel import SceneParameter
from ...radprops._particles import ParticleProperties
from ...spectral.index import MonoSpectralIndex, SpectralIndex
from ...units import to_quantity, units_compatible
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


@define(eq=False, slots=False)
class ParticleStateComponent(ParticleComponent):
    """
    Base class for a ParticleComponent defined by a ``profile`` class and a
    ``properties`` class. The profile defines the variation of a set of
    ``state_variables`` and ``amount`` in space, used to interpolate the
    ``properties`` at each location. The ``properties`` defines the mapping
    from state variables to optical properties such as extinction, albedo and
    phase function. This class is typically used for complex particle
    properties varying in space, such as clouds. When evaluated, the
    ``profile`` is resampled onto the render grid.

    The ``amount`` is a special state variable treated separately. It defines
    the density of the current species at the location. It is used only for
    scaling the extinction.

    The class is concrete, although its default (empty) parameters do not
    allow to properly define an interpolation behavior. It is thus possible to
    instantiate the class for a profile and properties pair that is not
    explicitly supported by Eradiate by default. In this case, users must
    provide the mapping ``state_variables`` and the ``amount`` variable that
    correspond to the targeted interpolation scheme.

    No specific units are enforced: the state variables exposed by the profile
    can be any physical quantity, as long as units are compatible between the
    source (``properties``) and destination (``profile``) variables linked by
    the ``state_variables`` dict. The ``amount`` units constraint lies in its
    combination with the properties dataset's per-``amount`` ``.ext`` quantity:
    the product of their units must have the dimensionality of an inverse
    length, expressing an ``amount``-scaled extinction coefficient. These
    units compatibility requirements are only checked when the component is
    evaluated (e.g. when computing extinction, albedo or the phase function).

    Interpolation of the ``properties`` against ``state_variables`` is linear.
    Extinction scaling is linear. Non-linear behaviors must be described by the
    ``properties``, that is treated as a simple lookup table by this class.
    Spatial resampling of the profile onto the render grid is axis-dependent:
    the ``x`` and ``y`` grids are not interpolated and must match the render
    grid's extent and resolution; the ``z`` axis is resampled onto the render
    grid's layers using a nearest-neighbour filter.

    Extinction and albedo are interpolated per voxel in Python; the phase
    function is interpolated per voxel by the
    :class:`.GriddedParticlePhaseFunction` kernel plugin at render time.
    Exactly one or two state dimensions are currently supported.

    Notes
    -----
    * The profile may have an irregular z-grid, but its x and y grids must
      be regular and match the render grid's extent and resolution exactly.
    * The profile's ``mass_concentration`` is set to zero for any render-grid
      layer outside its extent.
    """

    @classmethod
    def convert(cls, value: Any) -> ParticleStateComponent:
        """
        Conversion logic:

        * If value is a ParticleStateComponent instance, return it unchanged.
        * If value is a dictionary, it is interpreted by
          :data:`.atmosphere_factory`; the ``"type"`` parameter may be
          omitted and will automatically be set to ``"particle_state"``.
        * Otherwise, the value is returned unchanged.
        """
        if isinstance(value, MutableMapping) and "type" not in value:
            value = {**value, "type": "particle_state"}
        return atmosphere_factory.convert(value, allowed_cls=cls)

    profile: ParticleProfile = documented(
        attrs.field(kw_only=True, converter=ParticleProfile.convert),
        doc="Dense volumetric state profile, in the "
        ":ref:`Pfd-Core v1 <sec-data_formats-pfd_core_v1>` format. "
        "This parameter has no default.",
        type="ParticleProfile",
        init_type="ParticleProfile or Dataset or path-like",
    )

    properties: ParticleProperties = documented(
        attrs.field(kw_only=True, converter=ParticleProperties.convert),
        doc="Particle properties, in the "
        ":ref:`Prt-psd v1 <sec-data_formats-prt_psd_v1>` format. "
        "This parameter has no default.",
        type="ParticleProperties",
        init_type="ParticleProperties or Dataset or path-like or str",
    )

    @properties.validator
    def _properties_validator(self, attribute, value):
        if len(self.state_dims) not in (1, 2):
            expected = (
                f" (expected {list(self.state_variables)})"
                if self.state_variables
                else ""
            )
            raise ValueError(
                "'properties' must have exactly 1 or 2 state dimensions, "
                f"got {self.state_dims!r}{expected}"
            )

        missing = [
            d for d in self.state_dims if self._binding[d] not in self.profile.variables
        ]
        if missing:
            raise ValueError(
                f"'profile' has no variable for state dimension(s) {missing} "
                f"(expected {[self._binding[d] for d in missing]})"
            )

    amount: str | None = documented(
        attrs.field(
            default=None,
            kw_only=True,
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Name of the profile variable that scales the per-unit-amount "
        "extinction (e.g. mass or number concentration). If unset, the "
        "unique profile variable whose units make ``amount * ext`` an "
        "inverse length is selected automatically.",
        type="str or None",
        init_type="str, optional",
        default="None",
    )

    state_variables: dict[str, str] = documented(
        attrs.field(
            factory=dict,
            kw_only=True,
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.instance_of(str),
                mapping_validator=attrs.validators.instance_of(dict),
            ),
        ),
        doc="Mapping from a properties table state dimension name to the "
        "profile variable supplying its per-voxel value. State dimensions "
        "not listed here default to a profile variable of the same name.",
        type="dict[str, str]",
        init_type="dict[str, str], optional",
        default="{}",
    )

    _phase: GriddedParticlePhaseFunction = None

    @property
    def state_dims(self) -> tuple[str, ...]:
        """
        Returns
        -------
        tuple[str, ...]
            Names of :attr:`properties`' state dimensions: its ``ext``
            variable's dimensions, excluding ``w``, ``phamat``, ``iangle``
            and ``imom``.
        """
        non_state = {"w", "phamat", "iangle", "imom"}
        return tuple(d for d in self.properties.data["ext"].dims if d not in non_state)

    @property
    def _binding(self) -> dict[str, str]:
        """
        Returns
        -------
        dict[str, str]
            Mapping from each of :attr:`state_dims` to the profile
            variable supplying its per-voxel value, defaulting to a
            same-named variable when not listed in :attr:`state_variables`.
        """
        return {d: self.state_variables.get(d, d) for d in self.state_dims}

    @property
    def _amount(self) -> str:
        """
        Returns
        -------
        str
            Name of the profile variable that scales the per-unit-amount
            extinction: :attr:`amount` if set, otherwise the unique
            profile variable whose units make ``amount * ext`` an
            inverse length.
        """
        if self.amount is not None:
            return self.amount

        ext_units = self.properties.ext.units
        exclude = set(self._binding.values())

        matches = [
            name
            for name in self.profile.variables
            if name not in exclude
            and units_compatible(self.profile[name].units * ext_units, 1 / ureg.m)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot select amount variable unambiguously: candidates {matches}"
            )
        return matches[0]

    @cache_by_id
    def _eval_ext_ssa_grid(
        self, si: SpectralIndex
    ) -> tuple[pint.Quantity, pint.Quantity]:
        """
        Evaluate ``ext``/``ssa`` at ``si.w``, over the whole state-variable
        grid (no per-voxel resolution). Axis order matches :attr:`state_dims`.
        """
        pp = self.properties
        w = np.atleast_1d(si.w)
        return pp.eval_ext(w)[0], pp.eval_ssa(w)[0]

    def _interp_state(
        self, grid: pint.Quantity, resampled: ParticleProfile
    ) -> pint.Quantity:
        """
        Interpolate a state-variable-indexed ``grid`` (as returned by
        :meth:`_eval_ext_ssa_grid`) at ``resampled``'s per-voxel state
        values, clipped to ``grid``'s coordinate range along each state
        dimension.
        """
        query = {d: resampled[self._binding[d]] for d in self.state_dims}
        return self.properties.interp(grid, query)

    @cache_by_id
    def _eval_phase_data(self, si: SpectralIndex) -> dict:
        """
        Build the dict expected by :class:`.GriddedParticlePhaseFunction`'s
        ``phase_data`` callable: ``mu``/``phase`` flattened to one row per
        state-dimension combination, combined with :meth:`_eval_ext_ssa_grid`.
        """
        state_shape = tuple(self.properties.data.sizes[d] for d in self.state_dims)
        n_pts = int(np.prod(state_shape)) if state_shape else 1

        mu, phase = self.properties.eval_phase(si.w)
        n_phamat, *_, max_len = phase.shape
        mu = mu.reshape(n_pts, max_len)
        phase = phase.reshape(n_phamat, n_pts, max_len).transpose(1, 0, 2)

        ext_grid, ssa_grid = self._eval_ext_ssa_grid(si)
        return {
            "mu": mu,
            "phase": phase,
            "ext": ext_grid,
            "ssa": ssa_grid,
            "max_union_size": self.properties.max_union_size,
        }

    @property
    def phase(self) -> GriddedParticlePhaseFunction:
        if self._phase:
            return self._phase

        grid = self.geometry.grid
        resampled = self.profile.resample_regular(grid)
        amount = resampled[self._amount]
        empty = amount.m <= 0

        volumes = []
        grids = []
        for d in self.state_dims:
            value = resampled[self._binding[d]]
            volume_zyx = np.where(empty, np.nan, value.m)
            volumes.append(np.transpose(volume_zyx, (2, 1, 0)) * value.units)
            grids.append(to_quantity(self.properties.data[d]))

        kwargs = {
            "geometry": self.geometry,
            "phase_data": self._eval_phase_data,
            "volume_1": volumes[0],
            "grid_1": grids[0],
        }
        if len(volumes) > 1:
            kwargs["volume_2"] = volumes[1]
            kwargs["grid_2"] = grids[1]

        return GriddedParticlePhaseFunction(**kwargs)

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        _, result = traverse(self.phase)
        return result.data

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate the extinction coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.

        grid : :class:`.SpatialGrid`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Extinction coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        if not (self.has_absorption or self.has_scattering):
            raise RuntimeError(
                "At least one of 'has_absorption' or 'has_scattering' must be True."
            )

        grid = grid or self.geometry.grid
        col_units = uck.get("collision_coefficient")
        resampled = self.profile.resample_regular(grid)
        ext_grid, _ = self._eval_ext_ssa_grid(si)

        amount = resampled[self._amount]
        m_extinction = self._interp_state(ext_grid, resampled)
        extinction = (m_extinction * amount).to(col_units)

        extinction_values = np.where(amount.m > 0, extinction.m, 0.0)
        sigma_t_values = np.transpose(extinction_values, (2, 1, 0))  # (z,y,x)->(x,y,z)
        return sigma_t_values * col_units

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate the single-scattering albedo on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.

        grid : :class:`.SpatialGrid`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Single-scattering albedo array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, dimensionless.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        if self.has_absorption and self.has_scattering:
            grid = grid or self.geometry.grid
            resampled = self.profile.resample_regular(grid)
            _, ssa_grid = self._eval_ext_ssa_grid(si)

            amount = resampled[self._amount]
            albedo = self._interp_state(ssa_grid, resampled)
            albedo_m = np.where(amount.m > 0, albedo.m, 0.0)

            albedo_values = np.transpose(albedo_m, (2, 1, 0))  # (z,y,x)->(x,y,z)
            return albedo_values * ureg.dimensionless

        if self.has_absorption:
            return ureg.Quantity(0.0, ureg.dimensionless)
        if self.has_scattering:
            return ureg.Quantity(1.0, ureg.dimensionless)

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate the scattering coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.

        grid : :class:`.SpatialGrid`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Scattering coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        units = uck.get("collision_coefficient")
        if not self.has_scattering:
            return ureg.Quantity(np.zeros(grid.shape, dtype=np.float64), units)
        return self.eval_sigma_t(si, grid) * self.eval_albedo(si, grid).m_as(
            ureg.dimensionless
        )

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate the absorption coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.

        grid : :class:`.SpatialGrid`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Absorption coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        if not self.has_absorption:
            units = uck.get("collision_coefficient")
            return ureg.Quantity(np.zeros(grid.shape, dtype=np.float64), units)
        return self.eval_sigma_t(si, grid) * (
            1.0 - self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        )

    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        # Inherit docstring
        if not (self.has_absorption or self.has_scattering):
            raise RuntimeError(
                "At least one of 'has_absorption' or 'has_scattering' must be True."
            )
        min_sigma_s = self.eval_sigma_s(ctx.si).min(axis=-1)
        with np.errstate(divide="ignore"):
            return 1.0 / min_sigma_s

    # --------------------------------------------------------------------------
    #                       Profile-aligned geometry
    # --------------------------------------------------------------------------

    def _profile_regular_grid(self) -> CartesianGrid:
        """
        Return a regular :class:`.CartesianGrid` aligned with
        and fine enough to resolve the profile's z-levels.
        """

        def _regularize(arr: np.ndarray) -> np.ndarray:
            """Resample ``arr`` to a regular grid at least as fine as its
            smallest gap, spanning the same extent."""
            min_gap = np.diff(arr).min()
            n = round((arr[-1] - arr[0]) / min_gap) * 2 + 1
            return np.linspace(arr[0], arr[-1], n)

        x_levels = self.profile.data["x_levels"].values * ureg.km
        y_levels = self.profile.data["y_levels"].values * ureg.km
        z_levels = _regularize(self.profile.data["z_levels"].values) * ureg.km

        return CartesianGrid(
            edges_x=x_levels,
            edges_y=y_levels,
            levels=z_levels,
        ).centered()

    def to_profile_regular_grid(self):
        """
        Return a copy of this :class:`ParticleStateComponent` with its geometry set to
        a regular grid aligned with the profile.
        """
        grid = self._profile_regular_grid()
        geometry = PlaneParallelGeometry(
            grid=grid,
            toa_altitude=grid.levels[-1],
            width=self.geometry.width,
        )
        return attrs.evolve(self, geometry=geometry)


@define(eq=False, slots=False)
class ParticlePSDComponent(ParticleStateComponent):
    """
    Atmospheric medium representing a particle component with spatially
    heterogeneous physical properties indexed by effective radius (``reff``)
    and variance (``veff``). The amount of the species is described by a
    mass concentration (``mass_concentration``).

    The particle component is defined by a dense volumetric *profile*
    (per-voxel microphysical properties, in the
    :ref:`Pfd-core v1 <sec-data_formats-pfd_core_v1>` format) and particle
    properties (in the :ref:`Prt-Psd v1 <sec-data_formats-prt_psd_v1>` format).
    """

    amount: str | None = documented(
        attrs.field(
            default="mass_concentration",
            kw_only=True,
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(ParticleStateComponent, "amount", "doc"),
        type=get_doc(ParticleStateComponent, "amount", "type"),
        init_type=get_doc(ParticleStateComponent, "amount", "init_type"),
        default='"mass_concentration"',
    )

    state_variables: dict[str, str] = documented(
        attrs.field(
            default={"reff": "reff", "veff": "veff"},
            kw_only=True,
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.instance_of(str),
                mapping_validator=attrs.validators.instance_of(dict),
            ),
        ),
        doc=get_doc(ParticleStateComponent, "state_variables", "doc"),
        type=get_doc(ParticleStateComponent, "state_variables", "type"),
        init_type=get_doc(ParticleStateComponent, "state_variables", "init_type"),
        default='{ "reff": "reff", "veff": "veff" }',
    )

    @classmethod
    def convert(cls, value: Any) -> ParticlePSDComponent:
        """
        Conversion logic:

        * If value is a ParticlePSDComponent instance, return it unchanged.
        * If value is a dictionary, it is interpreted by
          :data:`.atmosphere_factory`; the ``"type"`` parameter may be
          omitted and will automatically be set to ``"particle_psd"``.
        * Otherwise, the value is returned unchanged.
        """
        if isinstance(value, MutableMapping) and "type" not in value:
            value = {**value, "type": "particle_psd"}
        return atmosphere_factory.convert(value, allowed_cls=cls)

    @classmethod
    def from_particle_aot(
        cls,
        particle_aot: ParticleAOTComponent,
        mass_concentration: pint.Quantity,
        reff: pint.Quantity,
        veff: pint.Quantity,
        grid: SpatialGrid | None = None,
    ) -> ParticlePSDComponent:
        """
        Build a :class:`.ParticlePSDComponent` from a :class:`.ParticleAOTComponent`.

        Parameters
        ----------
        particle_aot: :class:`.ParticleAOTComponent`
            Source particle aot.

        mass_concentration : :class:`pint.Quantity`
            Mass concentration used to derive the mass extinction coefficient.

        reff : :class:`pint.Quantity`
            Effective radius of the particle species.

        veff : :class:`pint.Quantity`
            Effective variance (dimensionless) of the particle species.

        grid : :class:`.SpatialGrid`, optional
            Target render grid for the output :class:`.ParticlePSDComponent`.
            If provided, ``particle_aot``'s radiative properties are
            evaluated on it instead of its own geometry's grid.

        Returns
        -------
        :class:`.ParticlePSDComponent`
        """
        output_geometry = (
            particle_aot.geometry
            if grid is None
            else attrs.evolve(particle_aot.geometry, grid=grid)
        )
        target_grid = output_geometry.grid
        w_ref = particle_aot.w_ref
        pp = particle_aot.particle_properties

        sigma_t_at_wref = pp.eval_ext(w_ref).item()
        albedo_at_wref = pp.eval_ssa(w_ref).item()

        m_extinction = (sigma_t_at_wref / mass_concentration).to(
            str(sigma_t_at_wref.units / mass_concentration.units)
        )

        si_ref = MonoSpectralIndex(w=w_ref)
        sigma_t_grid = particle_aot.eval_sigma_t(si_ref, target_grid)
        sigma_t_grid = ureg.Quantity(
            np.broadcast_to(
                sigma_t_grid.m,
                (target_grid.n_cells_x, target_grid.n_cells_y, target_grid.n_cells_z),
            ),
            sigma_t_grid.units,
        )
        mass_concentration_grid = (sigma_t_grid / m_extinction).to(
            str(mass_concentration.units)
        )

        # (x, y, z) -> (z, y, x)
        mass_concentration_zyx = np.transpose(mass_concentration_grid.m, (2, 1, 0))
        reff_zyx = np.full_like(mass_concentration_zyx, reff.m_as(ureg.micron))
        veff_zyx = np.full_like(mass_concentration_zyx, veff.m_as(ureg.dimensionless))

        level_units = {"units": "kilometer"}
        profile_ds = xr.Dataset(
            data_vars={
                "reff": (["z", "y", "x"], reff_zyx, {"units": "micron"}),
                "veff": (["z", "y", "x"], veff_zyx, {"units": "dimensionless"}),
                "mass_concentration": (
                    ["z", "y", "x"],
                    mass_concentration_zyx,
                    {"units": str(mass_concentration_grid.units)},
                ),
            },
            coords={
                "x_levels": (
                    ["x_levels"],
                    target_grid.edges_x.m_as(ureg.kilometer),
                    level_units,
                ),
                "y_levels": (
                    ["y_levels"],
                    target_grid.edges_y.m_as(ureg.kilometer),
                    level_units,
                ),
                "z_levels": (
                    ["z_levels"],
                    target_grid.levels.m_as(ureg.kilometer),
                    level_units,
                ),
            },
        )

        w_idx = int(
            np.argmin(
                np.abs(pp.data["w"].values - w_ref.m_as(pp.data["w"].attrs["units"]))
            )
        )
        n_valid = int(pp.data["nangles"].values[w_idx])
        mu = pp.data["mu"].values[w_idx, :n_valid]
        theta = np.rad2deg(np.arccos(mu))
        sort_idx = np.argsort(theta)[::-1]
        theta_sorted = theta[sort_idx]
        phase_compact = pp.phase.values[:, :n_valid, w_idx][:, sort_idx]
        n_theta = len(theta_sorted)
        n_phamat = phase_compact.shape[0]

        properties = make_prt_psd_v1(
            w=np.atleast_1d(w_ref),
            phamat=list(pp.data["phamat"].values),
            mu=ureg.Quantity(
                np.cos(np.deg2rad(theta_sorted))[np.newaxis, np.newaxis, np.newaxis, :],
                "dimensionless",
            ),
            theta=ureg.Quantity(
                theta_sorted[np.newaxis, np.newaxis, np.newaxis, :], ureg.degree
            ),
            ext=m_extinction.reshape(1, 1, 1),
            ssa=albedo_at_wref.reshape(1, 1, 1),
            phase=ureg.Quantity(
                phase_compact.reshape(n_phamat, 1, 1, 1, n_theta), ureg.steradians**-1
            ),
            pmom=np.zeros((1, 1, 1, 1)),
            reff=np.atleast_1d(reff),
            veff=np.atleast_1d(veff),
        )

        return cls(
            profile=profile_ds,
            properties=properties,
            geometry=output_geometry,
            has_absorption=particle_aot.has_absorption,
            has_scattering=particle_aot.has_scattering,
        )
