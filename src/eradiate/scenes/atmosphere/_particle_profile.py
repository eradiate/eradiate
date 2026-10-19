"""
Particle profiles.

Particle profiles describe the spatial distribution of a particle field:
a set of named per-voxel variables (e.g. an amount and the state variables
indexing particle properties) on a regular grid.
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

_STRUCTURAL_VARIABLES = ("x_edges", "y_edges", "z_edges")


@define(slots=False)
class ParticleProfile:
    """
    Dense, per-voxel description of a spatially varying particle field, in
    the :ref:`Pfd-Core v1 <sec-data_formats-pfd_core_v1>` format.

    Variables are accessed by name with ``profile[name]``, which returns a
    cached :class:`pint.Quantity` of shape ``(n_z, n_y, n_x)``.
    """

    data: xr.Dataset = documented(
        attrs.field(validator=attrs.validators.instance_of(xr.Dataset)),
        doc="Particle profile dataset, in Pfd-Core v1 format.",
        type="xarray.Dataset",
        init_type="xarray.Dataset",
    )

    # Per-variable quantities, filled on initialization
    _quantities: dict[str, pint.Quantity] = attrs.field(
        factory=dict, init=False, repr=False
    )

    def __attrs_post_init__(self):
        self._quantities = {
            name: to_quantity(self.data[name]) for name in self.variables
        }

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
        missing = [name for name in _STRUCTURAL_VARIABLES if name not in value]
        if missing:
            raise ValueError(
                f"Particle profile dataset is missing required variable(s): {missing}"
            )

        variables = [v for v in value.data_vars if v not in _STRUCTURAL_VARIABLES]
        if not variables:
            raise ValueError(
                "Particle profile dataset has no variable besides "
                f"{list(_STRUCTURAL_VARIABLES)}"
            )

        wrong_dims = [v for v in variables if value[v].dims != ("z", "y", "x")]
        if wrong_dims:
            raise ValueError(
                "Particle profile variables must have dims ('z', 'y', 'x'); "
                f"offending variable(s): {wrong_dims}"
            )

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Returns
        -------
        tuple[str, ...]
            Names of this profile's non-structural variables (``x_edges``,
            ``y_edges`` and ``z_edges`` excluded).
        """
        return tuple(v for v in self.data.data_vars if v not in _STRUCTURAL_VARIABLES)

    def __getitem__(self, name: str) -> pint.Quantity:
        """Return the cached per-voxel quantity for a non-structural variable."""
        try:
            return self._quantities[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not a variable of this profile: {self.variables}"
            ) from None

    @staticmethod
    def _check_regular(name: str, values: pint.Quantity) -> None:
        """Raise if ``values`` isn't evenly spaced."""
        step = np.diff(values)
        if not np.allclose(step, step[0]):
            raise ValueError(f"'{name}' must be regularly spaced")

    def resample_regular(self, grid: SpatialGrid) -> ParticleProfile:
        """
        Resample the profile onto a regular grid.

        The profile's own ``x_edges``/``y_edges`` must already be regularly
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
            If the profile's ``x_edges``/``y_edges`` are not regularly
            spaced, or do not match ``grid``'s extent and resolution.
        """

        return self._resample_regular_impl(grid)

    @cache_by_id
    def _resample_regular_impl(self, grid: SpatialGrid) -> ParticleProfile:

        if isinstance(grid, CartesianGrid):
            x_edges = to_quantity(self.data["x_edges"])
            y_edges = to_quantity(self.data["y_edges"])
            x_grid_edges = grid.edges_x
            y_grid_edges = grid.edges_y
        elif isinstance(grid, SphericalGrid):
            x_edges = to_quantity(self.data["x_edges"])
            y_edges = to_quantity(self.data["y_edges"])
            x_grid_edges = grid.azimuths
            y_grid_edges = grid.colatitudes
        else:
            raise TypeError(f"unsupported grid type {type(grid).__name__}")

        self._check_regular("x_edges", x_edges)
        self._check_regular("y_edges", y_edges)

        x_edges_m = x_edges.m_as(x_grid_edges.units)
        y_edges_m = y_edges.m_as(y_grid_edges.units)

        if (
            x_edges_m.shape != x_grid_edges.shape
            or y_edges_m.shape != y_grid_edges.shape
            or not np.allclose(x_edges_m, x_grid_edges.m)
            or not np.allclose(y_edges_m, y_grid_edges.m)
        ):
            raise ValueError(
                "Profile 'x_edges'/'y_edges' do not match the target "
                "grid's extent and resolution."
            )

        return self._resample_z(grid, x_grid_edges, y_grid_edges)

    def _resample_z(
        self,
        grid: SpatialGrid,
        x_edges_out: pint.Quantity,
        y_edges_out: pint.Quantity,
    ) -> ParticleProfile:
        """
        Resample this profile onto ``grid``'s ``z`` layers. Each target layer
        takes the values of the source layer containing its centre; target
        layers whose centre lies outside the profile's extent get zero for
        every variable.

        Parameters
        ----------
        grid : .SpatialGrid
            Target grid providing ``z`` layer centres.
        x_edges_out, y_edges_out : :class:`pint.Quantity`
            Horizontal cell edges to store on the output profile, as-is.

        Returns
        -------
        ParticleProfile
            A new profile resampled onto ``grid``'s vertical layers.
        """
        z_edges_src = to_quantity(self.data["z_edges"]).m_as(ureg.meter)
        z_centers_tgt = grid.layers.m_as(ureg.meter)
        n_z_src = len(z_edges_src) - 1

        # Look up the source layer containing each target layer centre
        inn = np.clip(
            np.searchsorted(z_edges_src, z_centers_tgt, side="right") - 1,
            0,
            n_z_src - 1,
        )
        out_of_range = (z_centers_tgt < z_edges_src[0]) | (
            z_centers_tgt > z_edges_src[-1]
        )

        data_vars = {}
        for name in self.variables:
            value = self[name]  # (z, y, x)
            value_tgt = value.m[inn, :, :]
            value_tgt[out_of_range, :, :] = 0.0
            data_vars[name] = (
                ["z", "y", "x"],
                value_tgt,
                {"units": str(value.units)},
            )

        data_vars.update(
            {
                "x_edges": (
                    ["x_edges"],
                    x_edges_out.m,
                    {"units": str(x_edges_out.units)},
                ),
                "y_edges": (
                    ["y_edges"],
                    y_edges_out.m,
                    {"units": str(y_edges_out.units)},
                ),
                "z_edges": (
                    ["z_edges"],
                    grid.levels.m,
                    {"units": str(grid.levels.units)},
                ),
            }
        )

        return ParticleProfile(data=xr.Dataset(data_vars=data_vars))
