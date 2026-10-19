"""
Particle profiles.

Particle profiles describe the spatial distribution of a particle field:
the local particle size distribution parameters and mass concentration at
every voxel of a regular grid. Voxels with no particles carry a mass
concentration of zero.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import attrs
import numpy as np
import pint
import xarray as xr

from ...attrs import define, documented
from ...converters import load_dataset
from ...grid import CartesianGrid, SpatialGrid, SphericalGrid
from ...units import to_quantity
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


def _clip_and_interp(
    property_grid: xr.DataArray, reff_pts: xr.DataArray, veff_pts: xr.DataArray
) -> xr.DataArray:
    """Interpolate a ``(reff, veff)``-indexed property grid at *reff_pts*/*veff_pts*, clipped to its own coordinate range."""
    interp_coords = {}

    if property_grid.sizes.get("reff", 1) > 1:
        reff_vals = property_grid["reff"].values
        interp_coords["reff"] = reff_pts.clip(reff_vals.min(), reff_vals.max())
    elif "reff" in property_grid.dims:
        property_grid = property_grid.squeeze("reff")

    if property_grid.sizes.get("veff", 1) > 1:
        veff_vals = property_grid["veff"].values
        interp_coords["veff"] = veff_pts.clip(veff_vals.min(), veff_vals.max())
    elif "veff" in property_grid.dims:
        property_grid = property_grid.squeeze("veff")

    if interp_coords:
        property_grid = property_grid.interp(interp_coords, method="linear")

    return property_grid


@define(slots=False)
class ParticleProfile:
    """
    Dense, per-voxel description of a spatially varying particle field, in
    the :ref:`Pfd-Core v1 <sec-data_formats-pfd_core_v1>` format.
    """

    data: xr.Dataset = documented(
        attrs.field(validator=attrs.validators.instance_of(xr.Dataset)),
        doc="Particle profile dataset, in Pfd-Core v1 format.",
        type="xarray.Dataset",
        init_type="xarray.Dataset",
    )

    # The following fields are used to cache properties
    # -- Effective radius
    _reff: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Effective variance
    _veff: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    # -- Mass concentration
    _mass_concentration: pint.Quantity | None = attrs.field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        # Resolve all cached attributes
        for attr in ["reff", "veff", "mass_concentration"]:
            getattr(self, attr)

    @classmethod
    def convert(cls, value: Any) -> ParticleProfile:
        """
        Conversion logic:

        * If value is a ParticleProfile instance, return it.
        * If value is a Dataset, initialize a ParticleProfile with it.
        * If value is a path-like, load the dataset and initialize a
          ParticleProfile with it.
        """
        if isinstance(value, ParticleProfile):
            return value

        if isinstance(value, xr.Dataset):
            return ParticleProfile(value)

        if isinstance(value, (str, PathLike)):
            return ParticleProfile(load_dataset(value))

        raise TypeError(
            f"could not convert value of type {type(value).__name__} to ParticleProfile"
        )

    @data.validator
    def _validate_data(self, attribute, value):
        missing = [
            name
            for name in (
                "reff",
                "veff",
                "mass_concentration",
                "x_levels",
                "y_levels",
                "z_levels",
            )
            if name not in value
        ]
        if missing:
            raise ValueError(
                f"Particle profile dataset is missing required variable(s): {missing}"
            )

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Returns
        -------
        tuple[str, ...]
            Names of this profile's non-structural variables (``x_levels``,
            ``y_levels`` and ``z_levels`` excluded).
        """
        structural = {"x_levels", "y_levels", "z_levels"}
        return tuple(v for v in self.data.data_vars if v not in structural)

    @property
    def reff(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Effective radius grid, cached to minimize overhead.
        """
        if self._reff is None:
            self._reff = to_quantity(self.data["reff"])
        return self._reff

    @property
    def veff(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Effective variance grid, cached to minimize overhead.
        """
        if self._veff is None:
            self._veff = to_quantity(self.data["veff"])
        return self._veff

    @property
    def mass_concentration(self) -> pint.Quantity:
        """
        Returns
        -------
        pint.Quantity
            Mass concentration grid, cached to minimize overhead.
        """
        if self._mass_concentration is None:
            self._mass_concentration = to_quantity(self.data["mass_concentration"])
        return self._mass_concentration

    def __getitem__(self, name: str) -> pint.Quantity:
        """Return the per-voxel quantity for a non-structural variable."""
        if name not in self.variables:
            raise KeyError(
                f"'{name}' is not a variable of this profile: {self.variables}"
            )
        return getattr(self, name)

    @staticmethod
    def _check_regular(name: str, values: pint.Quantity) -> None:
        """Raise if ``values`` isn't evenly spaced."""
        step = np.diff(values)
        if not np.allclose(step, step[0]):
            raise ValueError(f"'{name}' must be regularly spaced")

    def resample_regular(self, grid: SpatialGrid) -> ParticleProfile:
        """
        Resample the profile onto a regular grid.

        The profile's own ``x_levels``/``y_levels`` must already be regularly
        spaced and must match ``grid``'s extent and resolution exactly. Each
        ``grid`` layer takes the values of the profile layer containing its
        centre.

        Parameters
        ----------
        grid : .SpatialGrid
            Target regular grid.

        Returns
        -------
        ParticleProfile
            A new profile, regular on all three axes.

        Raises
        ------
        ValueError
            If the profile's ``x_levels``/``y_levels`` are not regularly
            spaced, or do not match ``grid``'s extent and resolution.
        """

        return self._resample_regular_impl(grid)

    @cache_by_id
    def _resample_regular_impl(self, grid: SpatialGrid) -> ParticleProfile:

        if isinstance(grid, CartesianGrid):
            x_levels = to_quantity(self.data["x_levels"])
            y_levels = to_quantity(self.data["y_levels"])
            x_grid_edges = grid.edges_x
            y_grid_edges = grid.edges_y
        elif isinstance(grid, SphericalGrid):
            x_levels = to_quantity(self.data["x_levels"])
            y_levels = to_quantity(self.data["y_levels"])
            x_grid_edges = grid.azimuths
            y_grid_edges = grid.colatitudes
        else:
            raise TypeError(f"unsupported grid type {type(grid).__name__}")

        self._check_regular("x_levels", x_levels)
        self._check_regular("y_levels", y_levels)

        x_levels_m = x_levels.m_as(x_grid_edges.units)
        y_levels_m = y_levels.m_as(y_grid_edges.units)

        if (
            x_levels_m.shape != x_grid_edges.shape
            or y_levels_m.shape != y_grid_edges.shape
            or not np.allclose(x_levels_m, x_grid_edges.m)
            or not np.allclose(y_levels_m, y_grid_edges.m)
        ):
            raise ValueError(
                "Profile 'x_levels'/'y_levels' do not match the target "
                "grid's extent and resolution."
            )

        return self._resample_z(grid, x_grid_edges, y_grid_edges)

    def _resample_z(
        self,
        grid: SpatialGrid,
        x_levels_out: ping.Quantity,
        y_levels_out: pint.Quantity,
    ) -> ParticleProfile:
        """
        Resample this profile onto ``grid``'s ``z`` layers. Each target layer
        takes the values of the source layer containing its centre; target
        layers whose centre lies outside the profile's extent get a zero mass
        concentration.

        Parameters
        ----------
        grid : .SpatialGrid
            Target grid providing ``z`` layer centres.
        x_levels_out, y_levels_out : :class:`pint.Quantity`
            Horizontal level edges to store on the output profile, as-is.

        Returns
        -------
        ParticleProfile
            A new profile resampled onto ``grid``'s vertical layers.
        """
        z_levels_src = to_quantity(self.data["z_levels"]).m_as(ureg.meter)
        z_centers_tgt = grid.layers.m_as(ureg.meter)
        n_z_src = len(z_levels_src) - 1

        mass_concentration_unit = str(self.mass_concentration.units)
        reff_unit = str(self.reff.units)
        veff_unit = str(self.veff.units)

        reff_src = self.reff.m  # (z, y, x)
        veff_src = self.veff.m  # (z, y, x)
        mass_concentration_src = self.mass_concentration.m  # (z, y, x)

        # Look up the source layer containing each target layer centre
        inn = np.clip(
            np.searchsorted(z_levels_src, z_centers_tgt, side="right") - 1,
            0,
            n_z_src - 1,
        )
        in_range = (z_centers_tgt >= z_levels_src[0]) & (
            z_centers_tgt <= z_levels_src[-1]
        )

        reff_tgt = reff_src[inn, :, :]
        veff_tgt = veff_src[inn, :, :]
        mass_concentration_tgt = mass_concentration_src[inn, :, :]

        out_of_range = ~in_range
        if np.any(out_of_range):
            mass_concentration_tgt = mass_concentration_tgt.copy()
            mass_concentration_tgt[out_of_range, :, :] = 0.0

        data_vars = {
            "reff": (["z", "y", "x"], reff_tgt, {"units": reff_unit}),
            "veff": (["z", "y", "x"], veff_tgt, {"units": veff_unit}),
            "mass_concentration": (
                ["z", "y", "x"],
                mass_concentration_tgt,
                {"units": mass_concentration_unit},
            ),
            "x_levels": (
                ["x_levels"],
                x_levels_out.m,
                {"units": str(x_levels_out.units)},
            ),
            "y_levels": (
                ["y_levels"],
                y_levels_out.m,
                {"units": str(y_levels_out.units)},
            ),
            "z_levels": (
                ["z_levels"],
                grid.levels.m,
                {"units": str(grid.levels.units)},
            ),
        }

        return ParticleProfile(data=xr.Dataset(data_vars=data_vars))

    def interp_reff_veff(self, property_grid: xr.DataArray) -> xr.DataArray:
        """
        Continuously interpolate a ``(reff, veff)``-indexed property grid
        (e.g. an ext/ssa grid from :class:`.ParticleProperties`) at this
        profile's own per-voxel ``reff``/``veff`` (dims ``z``, ``y``, ``x``).

        Points are clipped to *property_grid*'s own ``reff``/``veff``
        coordinate range, so that voxels sitting exactly on (or just past,
        from floating-point drift) the properties grid boundary do not
        turn into NaNs.
        """
        reff_grid = to_quantity(property_grid["reff"])
        veff_grid = to_quantity(property_grid["veff"])
        reff_pts = xr.DataArray(
            self.reff.m_as(reff_grid.units),
            dims=["z", "y", "x"],
            attrs={"units": str(reff_grid.units)},
        )
        veff_pts = xr.DataArray(
            self.veff.m_as(veff_grid.units),
            dims=["z", "y", "x"],
            attrs={"units": str(veff_grid.units)},
        )
        return _clip_and_interp(property_grid, reff_pts, veff_pts)

    def to_reff_veff_volumes(self) -> tuple[pint.Quantity, pint.Quantity]:
        """
        Return this profile's per-voxel ``reff``/``veff`` as dense
        ``(n_x, n_y, n_z)`` volumes. NaN marks empty cells.
        """
        empty = self.mass_concentration.m <= 0  # (z, y, x)

        reff_zyx = np.where(empty, np.nan, self.reff.m)
        veff_zyx = np.where(empty, np.nan, self.veff.m)

        reff_volume = np.transpose(reff_zyx, (2, 1, 0))
        veff_volume = np.transpose(veff_zyx, (2, 1, 0))

        return (
            ureg.Quantity(reff_volume, self.reff.units),
            ureg.Quantity(veff_volume, self.veff.units),
        )
