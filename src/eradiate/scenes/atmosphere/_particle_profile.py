"""
Particle profiles.

Particle profiles describe the spatial variations of the state variables of
a particulate medium (e.g. an amount and the variables indexing particle
properties), each as a field on a voxel grid.
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
from ...units import to_quantity, units_compatible
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id

_EDGE_COORDS = {"x": "x_edges", "y": "y_edges", "z": "z_edges"}


@define(slots=False)
class ParticleProfile:
    """
    Dense, per-voxel description of the :term:`state variables <state
    variable>` of a particulate medium (:term:`amount` included), in the
    :ref:`Ppr-Core v1 <sec-data_formats-ppr_core_v1>` format.

    State variables are accessed by name with ``profile[name]``, which
    returns a cached :class:`pint.Quantity` of shape ``(n_z, n_y, n_x)``.
    """

    data: xr.Dataset = documented(
        attrs.field(validator=attrs.validators.instance_of(xr.Dataset)),
        doc="Particle profile dataset, in Ppr-Core v1 format.",
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
        * If value is a string or a path-like, interpret it as a file path,
          load the dataset and initialize a ParticleProfile with it.
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, xr.Dataset):
            return cls(value)

        if isinstance(value, (str, PathLike)):
            return cls(load_dataset(value))

        raise TypeError(
            f"could not convert value of type {type(value).__name__} to ParticleProfile"
        )

    @data.validator
    def _validate_data(self, attribute, value):
        missing = [name for name in _EDGE_COORDS.values() if name not in value.coords]
        if missing:
            raise ValueError(
                f"particle profile dataset is missing required coordinate(s): {missing}"
            )

        if not value.data_vars:
            raise ValueError("particle profile dataset has no state variable")

        wrong_dims = [v for v in value.data_vars if value[v].dims != ("z", "y", "x")]
        if wrong_dims:
            raise ValueError(
                "particle profile state variables must have dims ('z', 'y', 'x'); "
                f"offending variable(s): {wrong_dims}"
            )

        units = {}
        for dim, name in _EDGE_COORDS.items():
            edges = value[name]
            expected = value.sizes[dim] + 1
            if edges.dims != (name,) or edges.size != expected:
                raise ValueError(
                    f"particle profile coordinate '{name}' must have dims "
                    f"('{name}',) and size len({dim}) + 1 = {expected}, "
                    f"got dims {edges.dims} and size {edges.size}"
                )
            if "units" not in edges.attrs:
                raise ValueError(f"particle profile coordinate '{name}' has no units")
            if not np.all(np.diff(edges.values) > 0):
                raise ValueError(
                    f"particle profile coordinate '{name}' must be strictly increasing"
                )
            units[name] = ureg(edges.attrs["units"]).units

        if not units_compatible(units["z_edges"], ureg.m):
            raise ValueError(
                "particle profile coordinate 'z_edges' must have length units, "
                f"got '{units['z_edges']}'"
            )

        x_units, y_units = units["x_edges"], units["y_edges"]
        if not any(
            units_compatible(x_units, ref) and units_compatible(y_units, ref)
            for ref in (ureg.m, ureg.rad)
        ):
            raise ValueError(
                "particle profile coordinates 'x_edges'/'y_edges' must both have "
                f"length units or both angle units, got '{x_units}'/'{y_units}'"
            )

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Returns
        -------
        tuple[str, ...]
            Names of this profile's state variables, *i.e.* its data
            variables.
        """
        return tuple(self.data.data_vars)

    def __getitem__(self, name: str) -> pint.Quantity:
        """Return the cached per-voxel quantity for a state variable."""
        try:
            return self._quantities[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not a state variable of this profile "
                f"(available: {self.variables})"
            ) from None

    def resample(self, grid: SpatialGrid) -> ParticleProfile:
        """
        Resample the profile onto a render grid.

        Only the ``z`` axis is resampled: the profile's ``x_edges``/``y_edges``
        must match ``grid``'s extent and resolution exactly. Each ``grid``
        layer takes the values of the profile layer containing its centre;
        layers whose centre lies outside the profile's vertical extent get
        zero for every state variable.

        Parameters
        ----------
        grid : .SpatialGrid
            Target render grid.

        Returns
        -------
        ParticleProfile
            A new profile, regular on all three axes.

        Raises
        ------
        ValueError
            If the profile's ``x_edges``/``y_edges`` do not match ``grid``'s
            extent and resolution.

        TypeError
            If ``grid`` is neither a :class:`.CartesianGrid` nor a
            :class:`.SphericalGrid`.
        """

        return self._resample_impl(grid)

    @cache_by_id
    def _resample_impl(self, grid: SpatialGrid) -> ParticleProfile:
        # Grids are regular by construction, so an exact match with the target
        # edges also guarantees regular source edges
        if isinstance(grid, CartesianGrid):
            x_edges_tgt, y_edges_tgt = grid.x_edges, grid.y_edges
        elif isinstance(grid, SphericalGrid):
            x_edges_tgt, y_edges_tgt = grid.azimuths, grid.colatitudes
        else:
            raise TypeError(f"unsupported grid type {type(grid).__name__}")

        x_edges_src = to_quantity(self.data["x_edges"]).m_as(x_edges_tgt.units)
        y_edges_src = to_quantity(self.data["y_edges"]).m_as(y_edges_tgt.units)

        if (
            x_edges_src.shape != x_edges_tgt.shape
            or y_edges_src.shape != y_edges_tgt.shape
            or not np.allclose(x_edges_src, x_edges_tgt.m)
            or not np.allclose(y_edges_src, y_edges_tgt.m)
        ):
            raise ValueError(
                "profile 'x_edges'/'y_edges' do not match the target "
                "grid's extent and resolution"
            )

        return self._resample_z(grid, x_edges_tgt, y_edges_tgt)

    def _resample_z(
        self,
        grid: SpatialGrid,
        x_edges_tgt: pint.Quantity,
        y_edges_tgt: pint.Quantity,
    ) -> ParticleProfile:
        """
        Resample this profile onto ``grid``'s ``z`` layers. Each target layer
        takes the values of the source layer containing its centre; target
        layers whose centre lies outside the profile's extent get zero for
        every state variable.

        Parameters
        ----------
        grid : .SpatialGrid
            Target grid providing ``z`` layer centres.
        x_edges_tgt, y_edges_tgt : :class:`pint.Quantity`
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
        i_src = np.clip(
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
            value_tgt = value.m[i_src, :, :]
            value_tgt[out_of_range, :, :] = 0.0
            data_vars[name] = (
                ["z", "y", "x"],
                value_tgt,
                {"units": str(value.units)},
            )

        coords = {
            name: ([name], edges.m, {"units": str(edges.units)})
            for name, edges in [
                ("x_edges", x_edges_tgt),
                ("y_edges", y_edges_tgt),
                ("z_edges", grid.levels),
            ]
        }

        return ParticleProfile(data=xr.Dataset(data_vars=data_vars, coords=coords))
