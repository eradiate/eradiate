"""
Particle fields.

A particle field is an atmospheric medium whose optical properties vary
spatially, described by a sparse volumetric :class:`.ParticleProfile`
(voxel-indexed microphysical properties) and a Prt v1 particle properties
lookup dataset (wrapped in a :class:`.ParticleProperties`).
"""

from __future__ import annotations

import attrs
import numpy as np
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere
from ._particle_dist import ArrayParticleDistribution
from ._particle_ensemble import ParticleEnsemble
from ._particle_profile import ParticleProfile
from ..core import traverse
from ..geometry import PlaneParallelGeometry
from ..phase import ParticleFieldPhaseFunction
from ...attrs import define, documented
from ...data.convert import make_aer_core_v2
from ...grid import GridCoords, PlaneParallelGridCoords
from ...kernel import SceneParameter
from ...radprops._particles import ParticleProperties
from ...spectral.index import MonoSpectralIndex, SpectralIndex
from ...units import to_quantity
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id

PHAMAT_LABELS = {
    4: ["11", "12", "33", "34"],
    6: ["11", "12", "22", "33", "34", "44"],
}


@define(eq=False, slots=False)
class ParticleField(AbstractHeterogeneousAtmosphere):
    """
    Atmospheric medium representing a particle field with spatially heterogeneous
    physical properties.

    The particle field is defined by a sparse volumetric *profile* (voxel-indexed
    microphysical properties, in the :ref:`Ppr v1 <sec-data-formats-ppr_v1>`
    format) and a Prt v1 particle properties dataset.

    When evaluated, the profile is resampled onto the render grid.
    Extinction and albedo are interpolated per voxel in Python; the phase
    function is interpolated per voxel by the
    :class:`.ParticleFieldPhaseFunction` kernel plugin at render time.

    Notes
    -----
    * The profile may have an irregular z-grid, but its x and y grids must
      be regular.
    * The render grid must be contained within the z-extent of the profile.
    """

    profile: ParticleProfile = documented(
        attrs.field(kw_only=True, converter=ParticleProfile.convert),
        doc="Sparse volumetric cloud profile, in the "
        ":ref:`Ppr v1 <sec-data-formats-ppr_v1>` format. "
        "This parameter has no default.",
        type="ParticleProfile",
        init_type="ParticleProfile or Dataset or path-like",
    )

    properties: ParticleProperties = documented(
        attrs.field(kw_only=True, converter=ParticleProperties.convert),
        doc="Particle properties, in the "
        ":ref:`Prt v1 <sec-data-formats-prt_v1>` format. "
        "This parameter has no default.",
        type="ParticleProperties",
        init_type="ParticleProperties or Dataset or path-like or str",
    )

    @properties.validator
    def _properties_validator(self, attribute, value):
        if not value.has_size_distribution:
            raise ValueError(
                "While initialising ParticleField: 'properties' must have "
                "'reff'/'veff' dimensions (a Prt v1 dataset)"
            )

    phase_blending_method: str = documented(
        attrs.field(default="search", kw_only=True),
        doc="Phase blending method passed to the kernel plugin. "
        'Either ``"search"``, ``"stochastic"`` or ``"tabulate"``.',
        type="str",
        default='"search"',
    )

    has_absorption: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, the medium contributes an absorption coefficient.",
        type="bool",
        init_type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, the medium contributes a scattering coefficient.",
        type="bool",
        init_type="bool",
        default="True",
    )

    _phase: ParticleFieldPhaseFunction = None

    # --------------------------------------------------------------------------
    #                       Profile and properties access
    # --------------------------------------------------------------------------

    @cache_by_id
    def _eval_ext_ssa_grid(
        self, si: SpectralIndex
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Evaluate ``ext``/``ssa`` at ``si.w``, over the whole ``(reff, veff)``
        properties grid (no per-voxel resolution).
        """
        pp = self.properties
        w = np.atleast_1d(si.w)

        ext = pp.eval_ext(w)[0]
        ssa = pp.eval_ssa(w)[0]

        coords = {
            "reff": ("reff", pp.reff.m_as("micron"), {"units": "micron"}),
            "veff": (
                "veff",
                pp.veff.m_as(ureg.dimensionless),
                {"units": "dimensionless"},
            ),
        }
        ext_da = xr.DataArray(
            ext.m, dims=["reff", "veff"], coords=coords, attrs={"units": str(ext.units)}
        )
        ssa_da = xr.DataArray(
            ssa.m, dims=["reff", "veff"], coords=coords, attrs={"units": str(ssa.units)}
        )
        return ext_da, ssa_da

    @cache_by_id
    def _eval_phase_grid(
        self, si: SpectralIndex
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the phase function at ``si.w`` for every ``(reff, veff)``
        pair. Each pair's phase function is normalized independently, and
        pairs do not share the same angular discretization. Results are
        packed into NaN-padded arrays sized to the pair with the most angles.

        Returns
        -------
        mu : ndarray
            Shape ``(reff, veff, max angle count)``, NaN-padded.
        phase : ndarray
            Shape ``(phamat, reff, veff, max angle count)``, NaN-padded.
        nangles : ndarray
            Shape ``(reff, veff)``, valid angle count per pair.
        """
        return self.properties.eval_phase_grid(si.w)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_sigma_t(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the extinction coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
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
        resampled = self.profile.resample_regular(grid)
        ext_grid, _ = self._eval_ext_ssa_grid(si)

        ix = resampled.data["i_x"].values
        iy = resampled.data["i_y"].values
        iz = resampled.data["i_z"].values

        m_extinction = resampled.interp_reff_veff(ext_grid)
        extinction = (
            to_quantity(m_extinction)
            * to_quantity(resampled.data["mass_concentration"])
        ).to("1/km")

        sigma_t_values = np.zeros(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float64
        )
        sigma_t_values[ix, iy, iz] = extinction.m
        return ureg.Quantity(sigma_t_values, "1/km")

    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the single-scattering albedo on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
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

            ix = resampled.data["i_x"].values
            iy = resampled.data["i_y"].values
            iz = resampled.data["i_z"].values

            albedo_da = resampled.interp_reff_veff(ssa_grid)

            albedo_values = np.zeros(
                (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float64
            )
            albedo_values[ix, iy, iz] = to_quantity(albedo_da).m
            return ureg.Quantity(albedo_values, "dimensionless")

        if self.has_absorption:
            return ureg.Quantity(0.0, "dimensionless")
        if self.has_scattering:
            return ureg.Quantity(1.0, "dimensionless")

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the scattering coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Scattering coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        if not self.has_scattering:
            return ureg.Quantity(np.zeros(grid.shape, dtype=np.float64), "1/km")
        return self.eval_sigma_t(si, grid) * self.eval_albedo(si, grid).m_as(
            ureg.dimensionless
        )

    def eval_sigma_a(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the absorption coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
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
            return ureg.Quantity(np.zeros(grid.shape, dtype=np.float64), "1/km")
        return self.eval_sigma_t(si, grid) * (
            1.0 - self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        )

    def eval_mfp(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the mean free path on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Mean free path array of shape ``(n_cells_x, n_cells_y, n_cells_z)``,
            in metres.
        """
        grid = grid or self.geometry.grid
        sigma_t = self.eval_sigma_t(si, grid)
        sigma_t_values = sigma_t.m_as("1/m")
        mfp_values = np.full(sigma_t_values.shape, np.inf, dtype=np.float64)
        np.divide(1.0, sigma_t_values, where=sigma_t_values != 0, out=mfp_values)
        return ureg.Quantity(mfp_values, "m")

    def _eval_phase_data(
        self, si: SpectralIndex
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine :meth:`_eval_phase_grid` and :meth:`_eval_ext_ssa_grid`'s
        results into the ``(mu, phase, nangles, ext, ssa)`` tuple expected
        by :class:`.ParticleFieldPhaseFunction`'s ``phase_data`` callable.
        """
        mu, phase, nangles = self._eval_phase_grid(si)
        ext_grid, ssa_grid = self._eval_ext_ssa_grid(si)
        return mu, phase, nangles, ext_grid.values, ssa_grid.values

    @property
    def phase(self) -> ParticleFieldPhaseFunction:
        if self._phase:
            return self._phase

        grid = self.geometry.grid
        reff_volume, veff_volume = self.profile.resample_regular(
            grid
        ).to_reff_veff_volumes()
        pp = self.properties
        return ParticleFieldPhaseFunction(
            geometry=self.geometry,
            r_eff_volume=lambda ctx: reff_volume,
            v_eff_volume=lambda ctx: veff_volume,
            r_eff_grid=pp.reff.m_as("micron").astype(np.float64),
            v_eff_grid=pp.veff.m_as(ureg.dimensionless).astype(np.float64),
            phase_data=self._eval_phase_data,
            blending_method=self.phase_blending_method,
        )

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        _, result = traverse(self.phase)
        return result.data

    # --------------------------------------------------------------------------
    #                       Profile-aligned geometry
    # --------------------------------------------------------------------------

    def _profile_regular_grid(self) -> PlaneParallelGridCoords:
        """
        Return a regular :class:`.PlaneParallelGridCoords` aligned with
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

        return PlaneParallelGridCoords(
            edges_x=x_levels,
            edges_y=y_levels,
            levels=z_levels,
        ).centered()

    def to_profile_regular_grid(self):
        """
        Return a copy of this :class:`ParticleField` with its geometry set to
        a regular grid aligned with the profile.
        """
        grid = self._profile_regular_grid()
        geometry = PlaneParallelGeometry(
            grid=grid,
            toa_altitude=grid.levels[-1],
            width=self.geometry.width,
        )
        return attrs.evolve(self, geometry=geometry)

    # --------------------------------------------------------------------------
    #                       Voxel <-> ParticleEnsemble conversion
    # --------------------------------------------------------------------------

    @classmethod
    def to_particle_ensemble_from_profile_point(
        cls,
        particle_field: ParticleField,
        profile_point: xr.Dataset,
        w: ureg.Quantity,
        grid: GridCoords | None = None,
    ) -> ParticleEnsemble:
        """
        Build a :class:`.ParticleEnsemble` from a single profile point.

        Parameters
        ----------
        particle_field : :class:`.ParticleField`
            Source particle field.
        profile_point : xr.Dataset
            A single-point subset of a resampled profile dataset, e.g.
            obtained via
            ``particle_field.profile.resample_regular(...).data.sel(index=0)``.
            Must contain exactly one point along the ``index`` dimension.
        w : :class:`pint.Quantity`
            Evaluation wavelength.
        grid : :class:`.GridCoords`, optional
            Target render grid. Defaults to ``particle_field.geometry.grid``.

        Returns
        -------
        :class:`.ParticleEnsemble`
        """
        if profile_point.sizes.get("index", 1) != 1:
            raise ValueError(
                "profile_point must contain exactly one point along the 'index' dimension."
            )

        grid = grid or particle_field.geometry.grid
        wavelength = np.atleast_1d(w)
        pp = particle_field.properties

        ix = int(profile_point["i_x"].values.item())
        iy = int(profile_point["i_y"].values.item())
        iz = int(profile_point["i_z"].values.item())
        reff_qty = to_quantity(profile_point["reff"]).item()
        veff_qty = to_quantity(profile_point["veff"]).item()
        mass_concentration_quantity = to_quantity(
            profile_point["mass_concentration"]
        ).item()

        i_reff = int(
            np.argmin(np.abs(pp.reff.m_as("micron") - reff_qty.m_as("micron")))
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
        sigma_t = (m_extinction * mass_concentration_quantity).to("1/km")

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
        tau_ref = ureg.Quantity(
            (sigma_t * layer_height).m_as(ureg.dimensionless) * overlap,
            "dimensionless",
        )

        phase_ds = pp.phase_dataset_for_reff_veff(
            wavelength[0], i_reff, i_veff, sigma_t
        )

        return ParticleEnsemble(
            bottom=grid.levels[iz],
            top=grid.levels[iz + 1],
            distribution="uniform",
            w_ref=wavelength[0],
            tau_ref=tau_ref,
            particle_properties=phase_ds,
            geometry=particle_field.geometry,
        )

    @classmethod
    def to_particle_ensemble(
        cls,
        particle_field: ParticleField,
        w: ureg.Quantity,
        ix: int,
        iy: int,
        iz: int,
        grid: GridCoords | None = None,
    ) -> ParticleEnsemble:
        """
        Build a :class:`.ParticleEnsemble` representing a single voxel of the
        particle field at the given grid location.

        Parameters
        ----------
        particle_field : :class:`.ParticleField`
            Source particle field.
        w : :class:`pint.Quantity`
            Evaluation wavelength.
        ix, iy, iz : int
            Voxel indices in the render grid.
        grid : :class:`.GridCoords`, optional
            Target render grid. Defaults to ``particle_field.geometry.grid``.

        Returns
        -------
        :class:`.ParticleEnsemble`
        """
        grid = grid or particle_field.geometry.grid

        resampled_profile = particle_field.profile.resample_regular(grid).data

        voxel_mask = (
            (resampled_profile["i_x"].values == ix)
            & (resampled_profile["i_y"].values == iy)
            & (resampled_profile["i_z"].values == iz)
        )
        if not np.any(voxel_mask):
            raise ValueError(f"No cloud voxel found at ({ix}, {iy}, {iz}).")

        index = int(np.where(voxel_mask)[0][0])
        profile_point = resampled_profile.isel(index=index)

        return cls.to_particle_ensemble_from_profile_point(
            particle_field, profile_point, w, grid
        )

    @classmethod
    def to_particle_ensemble_fixed_composition(
        cls,
        particle_field: ParticleField,
        w: ureg.Quantity,
        i_reff: int,
        i_veff: int,
        grid: GridCoords | None = None,
        ext: ureg.Quantity | None = None,
    ) -> ParticleEnsemble:
        """
        Build a single :class:`.ParticleEnsemble` approximating the whole
        field with one fixed ``(reff, veff)`` grid point.

        Every voxel's own ``reff``/``veff`` is ignored in favour of the
        given grid point; only its ``mass_concentration`` is used to build
        the per-voxel density (``tau_ref`` and ``distribution``).

        Parameters
        ----------
        particle_field : :class:`.ParticleField`
            Source particle field.
        w : :class:`pint.Quantity`
            Evaluation wavelength.
        i_reff, i_veff : int
            Index along ``particle_field.properties``'s ``reff``/``veff``
            dimensions, assumed constant across the whole field.
        grid : :class:`.GridCoords`, optional
            Target render grid. Defaults to ``particle_field.geometry.grid``.
        ext : :class:`pint.Quantity`, optional
            Extinction coefficient stored in the output ensemble's
            ``particle_properties``. Defaults to the native grid value at
            ``(i_reff, i_veff)``. Since that dataset only ever has one
            wavelength point, this value is only ever compared to itself
            (``ext(w) / ext(w_ref)``) and has no effect on the ensemble's
            radiative properties; overriding it is rarely needed.

        Returns
        -------
        :class:`.ParticleEnsemble`
        """
        grid = grid or particle_field.geometry.grid
        wavelength = np.atleast_1d(w)
        pp = particle_field.properties

        # Evaluate on the wavelength axis first, then pick this exact point.
        m_extinction = pp.eval_ext(wavelength)[0, i_reff, i_veff]

        resampled = particle_field.profile.resample_regular(grid)
        mass_concentration = to_quantity(resampled.data["mass_concentration"])
        sigma_t = m_extinction * mass_concentration

        ix = resampled.data["i_x"].values
        iy = resampled.data["i_y"].values
        iz = resampled.data["i_z"].values
        sigma_t_grid = np.zeros(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float64
        )
        sigma_t_grid[ix, iy, iz] = sigma_t.m_as("1/km")

        layer_height = np.diff(grid.levels.m_as(ureg.km))
        tau_per_voxel = sigma_t_grid * layer_height[np.newaxis, np.newaxis, :]
        tau_ref = ureg.Quantity(tau_per_voxel.sum(axis=-1), "dimensionless")

        z_coords = (grid.layers - grid.levels[0]) / (grid.levels[-1] - grid.levels[0])
        z_coords = np.broadcast_to(
            z_coords.m_as(ureg.dimensionless), tau_per_voxel.shape
        )
        distribution = ArrayParticleDistribution(
            values=tau_per_voxel, coords=z_coords, method="nearest"
        )

        phase_ds = pp.phase_dataset_for_reff_veff(wavelength[0], i_reff, i_veff, ext)

        geometry = (
            particle_field.geometry
            if grid is particle_field.geometry.grid
            else attrs.evolve(particle_field.geometry, grid=grid)
        )

        return ParticleEnsemble(
            bottom=grid.levels[0],
            top=grid.levels[-1],
            distribution=distribution,
            w_ref=wavelength[0],
            tau_ref=tau_ref,
            particle_properties=phase_ds,
            geometry=geometry,
            has_absorption=particle_field.has_absorption,
            has_scattering=particle_field.has_scattering,
        )

    @classmethod
    def from_particle_ensemble(
        cls,
        particle_ensemble: ParticleEnsemble,
        mass_concentration: ureg.Quantity,
        reff: ureg.Quantity,
        veff: ureg.Quantity,
        grid: GridCoords | None = None,
    ) -> ParticleField:
        """
        Build a :class:`.ParticleField` from a :class:`.ParticleEnsemble`.

        Parameters
        ----------
        particle_ensemble : :class:`.ParticleEnsemble`
            Source particle ensemble.
        mass_concentration : :class:`pint.Quantity`
            Mass concentration used to derive the mass extinction coefficient.
        reff : :class:`pint.Quantity`
            Effective radius of the particle species.
        veff : :class:`pint.Quantity`
            Effective variance (dimensionless) of the particle species.
        grid : :class:`.GridCoords`, optional
            Target render grid for the output :class:`.ParticleField`.
            If provided, ``particle_ensemble``'s radiative properties are
            evaluated on it instead of its own geometry's grid.

        Returns
        -------
        :class:`.ParticleField`
        """
        output_geometry = (
            particle_ensemble.geometry
            if grid is None
            else attrs.evolve(particle_ensemble.geometry, grid=grid)
        )
        target_grid = output_geometry.grid
        w_ref = particle_ensemble.w_ref
        pp = particle_ensemble.particle_properties

        sigma_t_at_wref = pp.eval_ext(w_ref).item()
        albedo_at_wref = pp.eval_ssa(w_ref).item()

        m_extinction = (sigma_t_at_wref / mass_concentration).to(
            str(sigma_t_at_wref.units / mass_concentration.units)
        )

        si_ref = MonoSpectralIndex(w=w_ref)
        sigma_t_grid = particle_ensemble.eval_sigma_t(si_ref, target_grid)
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

        valid_mask = mass_concentration_grid.m > 0
        ix_arr, iy_arr, iz_arr = np.where(valid_mask)

        level_units = {"units": "kilometer"}
        profile_ds = xr.Dataset(
            data_vars={
                "i_x": (["index"], ix_arr),
                "i_y": (["index"], iy_arr),
                "i_z": (["index"], iz_arr),
                "reff": (
                    ["index"],
                    np.full(len(ix_arr), reff.m_as("micron")),
                    {"units": "micron"},
                ),
                "veff": (
                    ["index"],
                    np.full(len(ix_arr), veff.m_as(ureg.dimensionless)),
                    {"units": "dimensionless"},
                ),
                "mass_concentration": (
                    ["index"],
                    mass_concentration_grid.m[ix_arr, iy_arr, iz_arr],
                    {"units": str(mass_concentration_grid.units)},
                ),
            },
            coords={
                "x_levels": (
                    ["x"],
                    target_grid.edges_x.m_as(ureg.kilometer),
                    level_units,
                ),
                "y_levels": (
                    ["y"],
                    target_grid.edges_y.m_as(ureg.kilometer),
                    level_units,
                ),
                "z_levels": (
                    ["z"],
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

        properties = make_aer_core_v2(
            w=np.atleast_1d(w_ref),
            phamat=list(pp.data["phamat"].values),
            mu=ureg.Quantity(
                np.cos(np.deg2rad(theta_sorted))[np.newaxis, np.newaxis, np.newaxis, :],
                "dimensionless",
            ),
            theta=ureg.Quantity(
                theta_sorted[np.newaxis, np.newaxis, np.newaxis, :], "degree"
            ),
            ext=m_extinction.reshape(1, 1, 1),
            ssa=albedo_at_wref.reshape(1, 1, 1),
            phase=ureg.Quantity(
                phase_compact.reshape(n_phamat, 1, 1, 1, n_theta), "1/sr"
            ),
            pmom=np.zeros((1, 1, 1, 1)),
            reff=np.atleast_1d(reff),
            veff=np.atleast_1d(veff),
        )

        return cls(
            profile=profile_ds,
            properties=properties,
            geometry=output_geometry,
            has_absorption=particle_ensemble.has_absorption,
            has_scattering=particle_ensemble.has_scattering,
        )
