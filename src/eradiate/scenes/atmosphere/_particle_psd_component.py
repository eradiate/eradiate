"""
Particle state and PSD components.

A particle state component is an atmospheric medium whose optical properties
vary spatially, described by a dense volumetric :class:`.ParticleProfile`
(state variable fields) and a Prt-Core v1 particle properties lookup dataset
extended with state dimensions (wrapped in a :class:`.ParticleProperties`).
A particle PSD component is the specialization whose state variables are the
parameters of a particle size distribution.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import attrs
import numpy as np
import pint
import xarray as xr

from ._core import atmosphere_factory
from ._particle_component import ParticleComponent, ParticleOTComponent
from ._particle_profile import ParticleProfile
from ..core import traverse
from ..geometry import PlaneParallelGeometry
from ..phase import (
    GriddedParticlePhaseFunction,
    PhaseFunction,
    phase_function_factory,
)
from ...attrs import define, documented, get_doc
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
    Generic class for a particulate medium parametrized by an :term:`amount`
    and one or two tabulated :term:`state variables <state variable>`, which
    vary as a function of space coordinates.

    The ``profile`` holds the :term:`fields <field>` of the state variables.
    The tabulated ones serve as a lookup pivot in the ``properties`` table,
    which maps :term:`state dimensions <state dimension>` to per-unit-amount
    optical properties (extinction, albedo, phase function).

    Generic by nature, this class carries metadata to specify the tabulated
    state variables: the ``state_variables`` mapping links ``properties``
    state dimension names to the profile variables supplying their per-voxel
    values. No specific units are enforced: the state variables exposed by the
    profile can be any physical quantity, as long as units are compatible
    between the state dimension (``properties``) and the profile variable
    (``profile``) linked by the ``state_variables`` dictionary.

    The ``amount`` field specifies the profile variable that scales the
    per-unit-amount extinction (*e.g.* mass or number concentration).
    Its units must be such that its product with the units of the
    per-unit-amount extinction in ``properties`` has the dimensionality of an
    inverse length.

    Interpolation of the ``properties`` against ``state_variables`` is linear,
    and out-of-range state values are clamped.
    Extinction scaling is linear. When evaluated, the ``profile`` is resampled
    onto the render grid. This spatial resampling is axis-dependent: the ``x``
    and ``y`` grids are not interpolated and must match the render grid's
    extent and resolution; each render grid cell takes the values of the
    profile cell containing its centre, and cells whose centre lies outside
    the profile's vertical extent get zero for every state variable.

    Extinction and albedo are interpolated per voxel in Python; the phase
    function is interpolated per voxel at render time by the
    ``particlephase`` kernel plugin, wrapped by
    :class:`.GriddedParticlePhaseFunction`.
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
        ":ref:`Ppr-Core v1 <sec-data_formats-ppr_core_v1>` format. "
        "This parameter has no default.",
        type="ParticleProfile",
        init_type="ParticleProfile or Dataset or path-like",
    )

    properties: ParticleProperties = documented(
        attrs.field(kw_only=True, converter=ParticleProperties.convert),
        doc="Particle properties, in the "
        ":ref:`Prt-Core v1 <sec-data_formats-prt_core_v1>` format extended "
        "with 1 or 2 state dimensions (see "
        ":ref:`sec-data_formats-prt_extensions`). "
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

        # Checked here rather than on 'amount', which subclasses redefine
        if self.amount is not None and self.amount not in self.profile.variables:
            raise ValueError(
                f"'profile' has no amount variable '{self.amount}' "
                f"(available: {list(self.profile.variables)})"
            )

    amount: str | None = documented(
        attrs.field(
            default=None,
            kw_only=True,
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc="Name of the profile variable that scales the per-unit-amount "
        "extinction (*e.g.* mass or number concentration). If unset, the "
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
        doc="Tabulated state variables, as a mapping from a ``properties`` "
        "state dimension name to the profile variable supplying its "
        "per-voxel value. State dimensions not listed here default to a "
        "profile variable of the same name.",
        type="dict[str, str]",
        init_type="dict[str, str], optional",
        default="{}",
    )

    _phase: PhaseFunction | None = documented(
        attrs.field(
            default=None,
            kw_only=True,
            converter=attrs.converters.optional(phase_function_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(PhaseFunction)
            ),
            repr=False,
        ),
        doc="Phase function override. If unset, a "
        ":class:`.GriddedParticlePhaseFunction` is built from :attr:`profile` "
        "and :attr:`properties`.",
        type=".PhaseFunction or None",
        init_type=".PhaseFunction or dict, optional",
        default="None",
    )

    @property
    def state_dims(self) -> tuple[str, ...]:
        """
        Returns
        -------
        tuple[str, ...]
            Names of :attr:`properties`' state dimensions (see
            :attr:`.ParticleProperties.state_dims`).
        """
        return self.properties.state_dims

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
        n_pts = int(np.prod(self.properties.state_shape))

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
    def phase(self) -> PhaseFunction:
        if self._phase is not None:
            return self._phase

        grid = self.geometry.grid
        resampled = self.profile.resample(grid)
        amount = resampled[self._amount]
        empty = amount.m <= 0

        volumes = []
        grids = []
        for d in self.state_dims:
            value = resampled[self._binding[d]]
            volume_zyx = np.where(empty, np.nan, value.m)
            volumes.append(np.transpose(volume_zyx, (2, 1, 0)) * value.units)
            grids.append(self.properties.state_coord(d))

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
            ``(n_x_cells, n_y_cells, n_z_cells)``, in units of
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
        resampled = self.profile.resample(grid)
        ext_grid, _ = self._eval_ext_ssa_grid(si)

        amount = resampled[self._amount]
        m_extinction = self._interp_state(ext_grid, resampled)
        extinction = (m_extinction * amount).to(col_units)

        extinction_values = np.where(amount.m > 0, extinction.m, 0.0)
        sigma_t_values = np.transpose(extinction_values, (2, 1, 0))  # (z,y,x)->(x,y,z)

        # Remove the disabled interaction from the extinction
        if not self.has_absorption:
            sigma_t_values = sigma_t_values * self._eval_ssa(si, grid)
        elif not self.has_scattering:
            sigma_t_values = sigma_t_values * (1.0 - self._eval_ssa(si, grid))

        return sigma_t_values * col_units

    def _eval_ssa(self, si: SpectralIndex, grid: SpatialGrid) -> np.ndarray:
        """
        Evaluate the particles' single-scattering albedo on ``grid``,
        regardless of :attr:`has_absorption` and :attr:`has_scattering`.
        Empty voxels are set to 0. Axis order is ``(x, y, z)``.
        """
        resampled = self.profile.resample(grid)
        _, ssa_grid = self._eval_ext_ssa_grid(si)
        amount = resampled[self._amount]
        albedo = self._interp_state(ssa_grid, resampled)
        albedo_m = np.where(amount.m > 0, albedo.m_as(ureg.dimensionless), 0.0)
        return np.transpose(albedo_m, (2, 1, 0))  # (z,y,x)->(x,y,z)

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
            ``(n_x_cells, n_y_cells, n_z_cells)``, dimensionless.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        if self.has_absorption and self.has_scattering:
            grid = grid or self.geometry.grid
            return self._eval_ssa(si, grid) * ureg.dimensionless

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
            ``(n_x_cells, n_y_cells, n_z_cells)``, in units of
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
            ``(n_x_cells, n_y_cells, n_z_cells)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        if not self.has_absorption:
            units = uck.get("collision_coefficient")
            return ureg.Quantity(np.zeros(grid.shape, dtype=np.float64), units)
        return self.eval_sigma_t(si, grid) * (
            1.0 - self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        )

    # --------------------------------------------------------------------------
    #                       Profile-aligned geometry
    # --------------------------------------------------------------------------

    def _profile_regular_grid(self) -> CartesianGrid:
        """
        Return a regular :class:`.CartesianGrid` aligned with
        and fine enough to resolve the profile's z edges.
        """

        def _regularize(arr: np.ndarray) -> np.ndarray:
            """Resample ``arr`` to a regular grid at least as fine as its
            smallest gap, spanning the same extent."""
            min_gap = np.diff(arr).min()
            n = round((arr[-1] - arr[0]) / min_gap) * 2 + 1
            return np.linspace(arr[0], arr[-1], n)

        x_edges = to_quantity(self.profile.data["x_edges"])
        y_edges = to_quantity(self.profile.data["y_edges"])
        z_edges = (
            _regularize(to_quantity(self.profile.data["z_edges"]).m_as(ureg.km))
            * ureg.km
        )

        return CartesianGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            levels=z_edges,
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
    Particulate medium parametrized by a particle size distribution (PSD)
    that varies as a function of space coordinates [``particle_psd``].

    This :class:`.ParticleStateComponent` specialization uses the effective
    radius (``reff``) and effective variance (``veff``) of the PSD as tabulated
    state variables, and the mass concentration (``mass_concentration``) as amount.
    With the default ``state_variables`` and ``amount``, the ``profile`` is
    expected in the :ref:`Ppr-PSD v1 <sec-data_formats-ppr_psd_v1>` format
    and the ``properties`` in the
    :ref:`Prt-PSD v1 <sec-data_formats-prt_psd_v1>` format. The
    per-unit-amount extinction ``ext`` in ``properties`` is then a mass
    extinction coefficient (*e.g.* in m²/g).

    Interpolation, extinction scaling and spatial resampling are those of
    :class:`.ParticleStateComponent`.
    """

    profile: ParticleProfile = documented(
        attrs.field(kw_only=True, converter=ParticleProfile.convert),
        doc="Dense volumetric state profile, in the "
        ":ref:`Ppr-PSD v1 <sec-data_formats-ppr_psd_v1>` format. "
        "This parameter has no default.",
        type=get_doc(ParticleStateComponent, "profile", "type"),
        init_type=get_doc(ParticleStateComponent, "profile", "init_type"),
    )

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
            factory=lambda: {"reff": "reff", "veff": "veff"},
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
        default='{"reff": "reff", "veff": "veff"}',
    )

    # Checked here rather than on 'properties', whose validator is inherited
    @state_variables.validator
    def _state_variables_validator(self, attribute, value):
        if self.state_dims != ("reff", "veff"):
            raise ValueError(
                "'properties' must have state dimensions ('reff', 'veff'), "
                f"got {self.state_dims!r}"
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
    def from_particle_ot(
        cls,
        particle_ot: ParticleOTComponent,
        mass_concentration: pint.Quantity,
        reff: pint.Quantity,
        veff: pint.Quantity,
        grid: SpatialGrid | None = None,
    ) -> ParticlePSDComponent:
        """
        Build a :class:`.ParticlePSDComponent` from a :class:`.ParticleOTComponent`.

        The output has a spatially uniform PSD and a mass concentration field
        that reproduces ``particle_ot``'s extinction coefficient at its
        reference wavelength. Its ``properties`` hold a single wavelength
        point (``particle_ot.w_ref``) and a single ``(reff, veff)`` point.

        Parameters
        ----------
        particle_ot : :class:`.ParticleOTComponent`
            Source particle component. The ``ext`` variable of its
            ``particle_properties`` must have the dimensionality of an
            inverse length.

        mass_concentration : :class:`pint.Quantity`
            Mass concentration to which ``particle_ot``'s ``ext`` corresponds;
            the mass extinction coefficient of the output ``properties`` is
            ``ext(w_ref) / mass_concentration``.

        reff : :class:`pint.Quantity`
            Effective radius of the PSD, assigned to every voxel.

        veff : :class:`pint.Quantity`
            Effective variance (dimensionless) of the PSD, assigned to every
            voxel.

        grid : :class:`.SpatialGrid`, optional
            Target render grid for the output :class:`.ParticlePSDComponent`.
            If provided, ``particle_ot``'s radiative properties are
            evaluated on it instead of its own geometry's grid.

        Returns
        -------
        :class:`.ParticlePSDComponent`
        """
        output_geometry = (
            particle_ot.geometry
            if grid is None
            else attrs.evolve(particle_ot.geometry, grid=grid)
        )
        target_grid = output_geometry.grid
        w_ref = particle_ot.w_ref
        pp = particle_ot.particle_properties

        sigma_t_at_wref = pp.eval_ext(w_ref).item()
        albedo_at_wref = pp.eval_ssa(w_ref).item()

        m_extinction = (sigma_t_at_wref / mass_concentration).to(
            str(sigma_t_at_wref.units / mass_concentration.units)
        )

        si_ref = MonoSpectralIndex(w=w_ref)
        sigma_t_grid = particle_ot.eval_sigma_t(si_ref, target_grid)
        sigma_t_grid = ureg.Quantity(
            np.broadcast_to(
                sigma_t_grid.m,
                (target_grid.n_x_cells, target_grid.n_y_cells, target_grid.n_z_cells),
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
                "x_edges": (
                    ["x_edges"],
                    target_grid.x_edges.m_as(ureg.kilometer),
                    level_units,
                ),
                "y_edges": (
                    ["y_edges"],
                    target_grid.y_edges.m_as(ureg.kilometer),
                    level_units,
                ),
                "z_edges": (
                    ["z_edges"],
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
            reff=np.atleast_1d(reff),
            veff=np.atleast_1d(veff),
        )

        return cls(
            profile=profile_ds,
            properties=properties,
            geometry=output_geometry,
            has_absorption=particle_ot.has_absorption,
            has_scattering=particle_ot.has_scattering,
        )
