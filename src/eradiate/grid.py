from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import attrs
import mitsuba as mi
import numpy as np
import pint
import xarray as xr
from pinttrs.util import ensure_units

from .attrs import documented, frozen
from .kernel.transform import map_unit_cube
from .units import to_quantity
from .units import unit_context_config as ucc
from .units import unit_context_kernel as uck
from .units import unit_registry as ureg
from .util import _repr_diagrams

#: Horizontal extent of single-column grids built by :meth:`.CartesianGrid.from_levels`.
_DEFAULT_HORIZONTAL_EXTENT = 1e6 * ureg.km


def _either(value: Any, alias: Any, name: str, alias_name: str) -> tuple[Any, str]:
    """
    Select exactly one of an argument and its alias.

    Returns
    -------
    value : object
        The value that is set.
    name : str
        Name of the argument that is set.

    Raises
    ------
    TypeError
        If both or neither are set.
    """
    if (value is None) == (alias is None):
        raise TypeError(f"exactly one of '{name}' or '{alias_name}' must be set")
    return (value, name) if value is not None else (alias, alias_name)


def _regular_edges(
    edges: pint.Quantity | xr.DataArray | np.typing.ArrayLike,
    default_units: pint.Unit,
    name: str,
) -> tuple[pint.Quantity, pint.Quantity, pint.Quantity]:
    """
    Validate a regularly spaced edge array and derive its cell centres and step.

    Parameters
    ----------
    edges
        Cell edge coordinates. Unitless values are interpreted in
        ``default_units``.
    default_units
        Units applied to unitless ``edges``.
    name
        Name used in error messages.

    Returns
    -------
    edges, centers, step : quantity
        Edges with units, cell centres and (scalar) cell size.

    Raises
    ------
    ValueError
        If units are incompatible with ``default_units``, if there are fewer
        than 2 edges or if they are not regularly spaced.
    """
    if isinstance(edges, xr.DataArray):
        edges = to_quantity(edges)
    edges = ensure_units(edges, default_units=default_units)
    if not edges.is_compatible_with(default_units):
        raise ValueError(
            f"{name} has units '{edges.units:~P}', expected units compatible "
            f"with '{default_units:~P}'"
        )
    if edges.size < 2:
        raise ValueError(f"{name} must have at least 2 values")
    step = np.diff(edges)
    if not np.allclose(step, step[0]):
        raise ValueError(f"{name} must be regularly spaced")
    return edges, edges[:-1] + 0.5 * step, step[0]


def _n_cells(
    extent: pint.Quantity,
    cell_size: pint.Quantity,
    atol: pint.Quantity | None,
) -> int:
    """
    Compute the number of cells of size ``cell_size`` fitting in ``extent``.

    Parameters
    ----------
    extent
        Length to fill.
    cell_size
        Cell size.
    atol
        Absolute tolerance on the remainder. If ``None``, the remainder is
        ignored.

    Returns
    -------
    int
        Number of whole cells fitting in ``extent``.

    Raises
    ------
    ValueError
        If ``atol`` is set and the remainder exceeds it.
    """
    ratio = (extent / cell_size).m_as(ureg.dimensionless)
    n = int(np.floor(ratio))
    remainder = (ratio - n) * cell_size
    if atol is not None and remainder > atol:
        raise ValueError(
            f"extent ({extent}) is not a multiple of the cell size "
            f"({cell_size}) within tolerance ({atol}); remainder: {remainder}"
        )
    return n


def _edges_field(axis: str) -> attrs.Attribute:
    return documented(
        attrs.field(),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc=f"Cell edges along the {axis} axis, regularly spaced. Unitless "
        "values are interpreted in the axis's default units (see the class "
        "description).",
    )


@frozen(eq=False, init=False)
class SpatialGrid(ABC):
    """
    Abstract base class for regular 3D grids on which atmospheric radiative
    properties are evaluated.

    A grid is defined by its cell edges along three axes X, Y and Z. All grids
    store the same data and expose it through axis-generic accessors
    (:attr:`x_edges`, :attr:`x_centers`, :attr:`x_cell_size`,
    :attr:`x_extent`, :attr:`n_x_cells`, etc.). Subclasses set the meaning
    and units of each axis and add aliases named after it:

    * :class:`.CartesianGrid` (plane-parallel geometry): local Cartesian frame
      (X: position [length], Y: position [length], Z: altitude [length]).
    * :class:`.SphericalGrid` (spherical-shell geometry): geocentric
      spherical frame (X: azimuth [angle], Y: colatitude [angle],
      Z: altitude [length]).

    The Z axis is altitude for all grids, with aliases :attr:`levels` (edges),
    :attr:`layers` (centres), :attr:`layer_height` (cell size) and
    :attr:`total_height` (extent).

    These frames locate objects in the scene; they are not geodetic
    coordinate systems or map projections.

    Edges are regularly spaced along each axis, with independent spacing per
    axis.

    Notes
    -----
    * Instances are immutable.
    * Instances are hashed by ID, so they can be passed to LRU-cached
      functions.
    * Instances are passed to the ``eval()`` family of methods of atmosphere
      components.
    """

    #: Unit context keys of the X, Y and Z axes.
    _AXIS_UNITS: ClassVar[tuple[str, str, str]]

    x_edges: pint.Quantity = _edges_field("X")
    y_edges: pint.Quantity = _edges_field("Y")
    z_edges: pint.Quantity = _edges_field("Z")
    _x_centers: pint.Quantity = attrs.field()
    _y_centers: pint.Quantity = attrs.field()
    _z_centers: pint.Quantity = attrs.field()
    _x_cell_size: pint.Quantity = attrs.field()
    _y_cell_size: pint.Quantity = attrs.field()
    _z_cell_size: pint.Quantity = attrs.field()

    def _init_edges(self, *axes: tuple[Any, str]) -> None:
        """
        Validate edges and initialize all fields.

        Parameters
        ----------
        *axes
            ``(edges, name)`` pairs for the X, Y and Z axes, where ``name`` is
            the argument name used in error messages.

        Raises
        ------
        TypeError
            If edges are missing along an axis.
        """
        kwargs = {}
        for axis, (value, name), units in zip("xyz", axes, self._AXIS_UNITS):
            if value is None:
                raise TypeError(f"'{name}' must be set")
            edges, centers, cell_size = _regular_edges(value, ucc.get(units), name)
            kwargs[f"{axis}_edges"] = edges
            kwargs[f"{axis}_centers"] = centers
            kwargs[f"{axis}_cell_size"] = cell_size
        self.__attrs_init__(**kwargs)

    @classmethod
    def convert(cls, value: Any) -> Any:
        """
        Convert a value to a :class:`.SpatialGrid`, if possible.

        Parameters
        ----------
        value
            Value to convert. Conversion rules:

            * ``"cartesian"``: return :meth:`.CartesianGrid.default`.
            * 1D quantity, array or list of numbers: interpret as altitude
              levels and return ``from_levels(value)``.
            * Sequence of three 1D quantities or arrays (or 2D array with
              three rows): interpret as ``(x_edges, y_edges, z_edges)`` and
              pass them to the constructor.
            * Dictionary: its ``"type"`` entry (``"cartesian"`` or
              ``"spherical"``) selects the grid class; other entries are
              passed to its constructor.
            * Anything else is returned unchanged.

            Array rules build an instance of the class this method is called
            on, or a :class:`.CartesianGrid` if it is called on
            :class:`.SpatialGrid`. Unitless arrays are interpreted in the
            default units of each axis.

        Returns
        -------
        SpatialGrid or object
            Converted grid, or ``value`` if no rule applies.

        Raises
        ------
        ValueError
            If ``value`` is a string other than ``"cartesian"`` or a dictionary
            with an unknown ``"type"``.
        """
        if isinstance(value, str):
            if value == "cartesian":
                return CartesianGrid.default()
            raise ValueError(
                f"cannot convert string {value!r} to a grid (expected 'cartesian')"
            )
        elif isinstance(value, dict):
            value = value.copy()
            grid_type = value.pop("type")
            grid_cls = {"cartesian": CartesianGrid, "spherical": SphericalGrid}.get(
                grid_type
            )
            if grid_cls is None:
                raise ValueError(
                    f"unknown grid type {grid_type!r} "
                    "(expected 'cartesian' or 'spherical')"
                )
            return grid_cls(**value)
        grid_cls = CartesianGrid if cls is SpatialGrid else cls
        if isinstance(value, list) and all(np.ndim(v) == 0 for v in value):
            value = np.asarray(value)
        if isinstance(value, (pint.Quantity, np.ndarray)) and value.ndim == 1:
            return grid_cls.from_levels(value)
        elif (
            isinstance(value, (tuple, list, pint.Quantity, np.ndarray))
            and len(value) == 3
            and all(np.ndim(v) == 1 for v in value)
        ):
            return grid_cls(*value)

        return value

    # --------------------------------------------------------------------------
    #                           Axis-generic accessors
    # --------------------------------------------------------------------------

    @property
    def x_centers(self) -> pint.Quantity:
        """Cell centres along the X axis."""
        return self._x_centers

    @property
    def y_centers(self) -> pint.Quantity:
        """Cell centres along the Y axis."""
        return self._y_centers

    @property
    def z_centers(self) -> pint.Quantity:
        """Cell centres along the Z axis."""
        return self._z_centers

    @property
    def x_cell_size(self) -> pint.Quantity:
        """Uniform cell size along the X axis."""
        return self._x_cell_size

    @property
    def y_cell_size(self) -> pint.Quantity:
        """Uniform cell size along the Y axis."""
        return self._y_cell_size

    @property
    def z_cell_size(self) -> pint.Quantity:
        """Uniform cell size along the Z axis."""
        return self._z_cell_size

    @property
    def x_extent(self) -> pint.Quantity:
        """Grid extent along the X axis."""
        return self.x_edges[-1] - self.x_edges[0]

    @property
    def y_extent(self) -> pint.Quantity:
        """Grid extent along the Y axis."""
        return self.y_edges[-1] - self.y_edges[0]

    @property
    def z_extent(self) -> pint.Quantity:
        """Grid extent along the Z axis."""
        return self.z_edges[-1] - self.z_edges[0]

    @property
    def n_x_cells(self) -> int:
        """Number of cells along the X axis."""
        return len(self._x_centers)

    @property
    def n_y_cells(self) -> int:
        """Number of cells along the Y axis."""
        return len(self._y_centers)

    @property
    def n_z_cells(self) -> int:
        """Number of cells along the Z axis."""
        return len(self._z_centers)

    @property
    def n_x_edges(self) -> int:
        """Number of edges along the X axis."""
        return len(self.x_edges)

    @property
    def n_y_edges(self) -> int:
        """Number of edges along the Y axis."""
        return len(self.y_edges)

    @property
    def n_z_edges(self) -> int:
        """Number of edges along the Z axis."""
        return len(self.z_edges)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of cells along each axis, in (X, Y, Z) order."""
        return (self.n_x_cells, self.n_y_cells, self.n_z_cells)

    @property
    def is_1d(self) -> bool:
        """``True`` if the grid has a single cell in X and Y."""
        return self.n_x_cells == 1 and self.n_y_cells == 1

    # --------------------------------------------------------------------------
    #                              Z-axis aliases
    # --------------------------------------------------------------------------

    @property
    def levels(self) -> pint.Quantity:
        """Altitude levels (alias of :attr:`z_edges`)."""
        return self.z_edges

    @property
    def layers(self) -> pint.Quantity:
        """Altitudes of layer centres (alias of :attr:`z_centers`)."""
        return self._z_centers

    @property
    def layer_height(self) -> pint.Quantity:
        """Uniform layer height (alias of :attr:`z_cell_size`)."""
        return self._z_cell_size

    @property
    def total_height(self) -> pint.Quantity:
        """Distance between the lowest and highest levels (alias of :attr:`z_extent`)."""
        return self.z_extent

    @property
    def n_levels(self) -> int:
        """Number of levels (alias of :attr:`n_z_edges`)."""
        return self.n_z_edges

    @property
    def n_layers(self) -> int:
        """Number of layers (alias of :attr:`n_z_cells`)."""
        return self.n_z_cells

    # --------------------------------------------------------------------------
    #                              Modified copies
    # --------------------------------------------------------------------------

    def with_cell_counts(
        self,
        n_x_cells: int | None = None,
        n_y_cells: int | None = None,
        n_z_cells: int | None = None,
    ) -> SpatialGrid:
        """
        Return a copy with the same bounds and new cell counts.

        Parameters
        ----------
        n_x_cells, n_y_cells, n_z_cells
            Number of cells along each axis. If ``None``, the axis is left
            unchanged.

        Returns
        -------
        SpatialGrid
            Grid of the same type as this one.
        """
        edges = []
        for axis, n in zip("xyz", (n_x_cells, n_y_cells, n_z_cells)):
            e = getattr(self, f"{axis}_edges")
            edges.append(e if n is None else np.linspace(e[0], e[-1], n + 1))
        return self.__class__(*edges)

    def with_cell_sizes(
        self,
        x_cell_size: pint.Quantity | float | None = None,
        y_cell_size: pint.Quantity | float | None = None,
        z_cell_size: pint.Quantity | float | None = None,
        x_atol: pint.Quantity | float | None = None,
        y_atol: pint.Quantity | float | None = None,
        z_atol: pint.Quantity | float | None = None,
    ) -> SpatialGrid:
        """
        Return a copy with the same bounds and new cell sizes.

        The cell count along each axis is the number of whole cells of the
        requested size fitting in the extent; the cell size is then adjusted so
        that the cells fill the extent exactly.

        Parameters
        ----------
        x_cell_size, y_cell_size, z_cell_size
            Target cell size along each axis. If ``None``, the axis is left
            unchanged. Unitless values are interpreted in the axis's default
            units.
        x_atol, y_atol, z_atol
            Absolute tolerance on the remainder left when dividing the extent
            by the cell size along each axis. If ``None``, the remainder is
            ignored. Unitless values are interpreted in the axis's default
            units.

        Returns
        -------
        SpatialGrid
            Grid of the same type as this one.

        Raises
        ------
        ValueError
            If a tolerance is set and the remainder along that axis exceeds it.
        """
        n_cells = []
        for axis, cell_size, atol, units in zip(
            "xyz",
            (x_cell_size, y_cell_size, z_cell_size),
            (x_atol, y_atol, z_atol),
            self._AXIS_UNITS,
        ):
            if cell_size is None:
                n_cells.append(None)
                continue
            units = ucc.get(units)
            cell_size = ensure_units(cell_size, default_units=units)
            if atol is not None:
                atol = ensure_units(atol, default_units=units)
            n_cells.append(_n_cells(getattr(self, f"{axis}_extent"), cell_size, atol))
        return self.with_cell_counts(*n_cells)

    # --------------------------------------------------------------------------
    #                                  Misc
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def to_world(self) -> mi.ScalarTransform4f:
        """
        Transform from the unit cube to the grid's volume, in kernel units,
        used as the ``to_world`` parameter of the kernel volume plugin.
        """

    def __rich_repr__(self):
        for axis in "xyz":
            for name in ["edges", "centers", "cell_size", "extent"]:
                yield f"{axis}_{name}", getattr(self, f"{axis}_{name}")
        yield "shape", self.shape

    def _repr_html_(self) -> str | None:
        return _repr_diagrams.grid_repr_html(self)


@frozen(eq=False, init=False)
class CartesianGrid(SpatialGrid):
    """
    Cartesian grid [``cartesian``].

    Grid for plane-parallel geometries, in a local Cartesian frame
    (X: position [length], Y: position [length], Z: altitude [length]).
    Unitless edges are interpreted in ``ucc['length']`` on all axes.

    The constructor takes cell edges along each axis; Z edges can
    alternatively be passed as ``levels`` (exclusive with ``z_edges``). The
    ``from_*`` class methods build grids centred horizontally at the origin
    from extents, cell sizes and cell counts; the ``with_*`` and ``*ed``
    methods return modified copies.
    """

    _AXIS_UNITS = ("length", "length", "length")

    def __init__(
        self,
        x_edges: pint.Quantity | xr.DataArray | None = None,
        y_edges: pint.Quantity | xr.DataArray | None = None,
        z_edges: pint.Quantity | xr.DataArray | None = None,
        *,
        levels: pint.Quantity | xr.DataArray | None = None,
    ):
        self._init_edges(
            (x_edges, "x_edges"),
            (y_edges, "y_edges"),
            _either(z_edges, levels, "z_edges", "levels"),
        )

    @property
    def to_world(self) -> mi.ScalarTransform4f:
        unit = uck.get("length")
        return map_unit_cube(
            self.x_edges.min().m_as(unit),
            self.x_edges.max().m_as(unit),
            self.y_edges.min().m_as(unit),
            self.y_edges.max().m_as(unit),
            self.z_edges.min().m_as(unit),
            self.z_edges.max().m_as(unit),
        )

    def __repr__(self) -> str:
        return (
            "<CartesianGrid "
            f"x={self.x_edges[[0, -1]]:~P} ({self.n_x_cells} cells) "
            f"y={self.y_edges[[0, -1]]:~P} ({self.n_y_cells} cells) "
            f"z={self.z_edges[[0, -1]]:~P} ({self.n_z_cells} layers)"
            ">"
        )

    @staticmethod
    def _centered_edges(extent: pint.Quantity, n_cells: int) -> pint.Quantity:
        """Return regularly spaced edges centred at zero."""
        half = extent / 2.0
        return np.linspace(-half, half, n_cells + 1)

    # --------------------------------------------------------------------------
    #                                 Factories
    # --------------------------------------------------------------------------

    @classmethod
    def default(cls) -> CartesianGrid:
        """
        Build the default grid for Eradiate atmospheres.

        Levels span 0 to 120 km with 100 m layers; the grid has a single
        horizontal cell spanning 10⁶ km in X and Y.

        Returns
        -------
        CartesianGrid
        """
        return cls.from_levels(np.linspace(0.0, 120.0, 1201) * ureg.km)

    @classmethod
    def from_levels(cls, levels: pint.Quantity | np.typing.ArrayLike) -> CartesianGrid:
        """
        Build a single-column grid from altitude levels.

        The single horizontal cell spans 10⁶ km in X and Y, centred at the
        origin.

        Parameters
        ----------
        levels
            Altitude levels (Z edges). Unitless values are interpreted in
            ``ucc['length']``.

        Returns
        -------
        CartesianGrid
        """
        return cls.from_extent_and_resolution(
            x_extent=_DEFAULT_HORIZONTAL_EXTENT,
            y_extent=_DEFAULT_HORIZONTAL_EXTENT,
            n_x_cells=1,
            n_y_cells=1,
            levels=levels,
        )

    @classmethod
    def from_extent_and_resolution(
        cls,
        x_extent: pint.Quantity | float,
        y_extent: pint.Quantity | float,
        n_x_cells: int,
        n_y_cells: int,
        z_edges: pint.Quantity | None = None,
        *,
        levels: pint.Quantity | None = None,
    ) -> CartesianGrid:
        """
        Build a grid centred horizontally at the origin from extents and cell
        counts.

        Parameters
        ----------
        x_extent, y_extent
            Grid extent along the X and Y axes. Unitless values are
            interpreted in ``ucc['length']``.
        n_x_cells, n_y_cells
            Number of cells along the X and Y axes.
        z_edges
            Cell edges along the Z axis. Unitless values are interpreted in
            ``ucc['length']``.
        levels
            Alias of ``z_edges`` (exclusive).

        Returns
        -------
        CartesianGrid
        """
        x_extent = ensure_units(x_extent, default_units=ucc.get("length"))
        y_extent = ensure_units(y_extent, default_units=ucc.get("length"))

        return cls(
            cls._centered_edges(x_extent, n_x_cells),
            cls._centered_edges(y_extent, n_y_cells),
            z_edges,
            levels=levels,
        )

    @classmethod
    def from_extent_and_cell_size(
        cls,
        x_extent: pint.Quantity | float,
        y_extent: pint.Quantity | float,
        x_cell_size: pint.Quantity | float,
        y_cell_size: pint.Quantity | float,
        z_edges: pint.Quantity | None = None,
        *,
        levels: pint.Quantity | None = None,
        x_atol: pint.Quantity | float | None = None,
        y_atol: pint.Quantity | float | None = None,
    ) -> CartesianGrid:
        """
        Build a grid centred horizontally at the origin from extents and cell
        sizes.

        The cell count along each axis is the number of whole cells of the
        requested size fitting in the extent; the cell size is then adjusted so
        that the cells fill the extent exactly.

        Parameters
        ----------
        x_extent, y_extent
            Grid extent along the X and Y axes. Unitless values are
            interpreted in ``ucc['length']``.
        x_cell_size, y_cell_size
            Cell size along the X and Y axes. Unitless values are interpreted
            in ``ucc['length']``.
        z_edges
            Cell edges along the Z axis. Unitless values are interpreted in
            ``ucc['length']``.
        levels
            Alias of ``z_edges`` (exclusive).
        x_atol, y_atol
            Absolute tolerance on the remainder left when dividing the extent
            by the cell size along the X and Y axes. If ``None``, the remainder
            is ignored. Unitless values are interpreted in ``ucc['length']``.

        Returns
        -------
        CartesianGrid

        Raises
        ------
        ValueError
            If a tolerance is set and the remainder along that axis exceeds it.
        """
        length = ucc.get("length")
        n_cells = []
        for extent, cell_size, atol in [
            (x_extent, x_cell_size, x_atol),
            (y_extent, y_cell_size, y_atol),
        ]:
            extent = ensure_units(extent, default_units=length)
            cell_size = ensure_units(cell_size, default_units=length)
            if atol is not None:
                atol = ensure_units(atol, default_units=length)
            n_cells.append(_n_cells(extent, cell_size, atol))

        return cls.from_extent_and_resolution(
            x_extent, y_extent, *n_cells, z_edges, levels=levels
        )

    @classmethod
    def from_cell_size_and_resolution(
        cls,
        x_cell_size: pint.Quantity | float,
        y_cell_size: pint.Quantity | float,
        n_x_cells: int,
        n_y_cells: int,
        z_edges: pint.Quantity | None = None,
        *,
        levels: pint.Quantity | None = None,
    ) -> CartesianGrid:
        """
        Build a grid centred horizontally at the origin from cell sizes and
        counts.

        The extent along each axis is ``cell_size * n_cells``.

        Parameters
        ----------
        x_cell_size, y_cell_size
            Cell size along the X and Y axes. Unitless values are interpreted
            in ``ucc['length']``.
        n_x_cells, n_y_cells
            Number of cells along the X and Y axes.
        z_edges
            Cell edges along the Z axis. Unitless values are interpreted in
            ``ucc['length']``.
        levels
            Alias of ``z_edges`` (exclusive).

        Returns
        -------
        CartesianGrid
        """
        x_cell_size = ensure_units(x_cell_size, default_units=ucc.get("length"))
        y_cell_size = ensure_units(y_cell_size, default_units=ucc.get("length"))

        return cls.from_extent_and_resolution(
            x_cell_size * n_x_cells,
            y_cell_size * n_y_cells,
            n_x_cells,
            n_y_cells,
            z_edges,
            levels=levels,
        )

    # --------------------------------------------------------------------------
    #                              Modified copies
    # --------------------------------------------------------------------------

    def extended_by_cells(self, n_x_cells: int, n_y_cells: int) -> CartesianGrid:
        """
        Return a copy extended symmetrically by whole cells.

        The cell size is preserved; the cell count grows by ``2 * n_x_cells``
        in X and ``2 * n_y_cells`` in Y.

        Parameters
        ----------
        n_x_cells, n_y_cells
            Number of cells added on each side along the X and Y axes.

        Returns
        -------
        CartesianGrid
        """

        def _extend(edges, step, n):
            prepend = edges[0] - step * np.arange(n, 0, -1)
            append = edges[-1] + step * np.arange(1, n + 1)
            return np.concatenate([prepend, edges, append])

        return self.__class__(
            _extend(self.x_edges, self.x_cell_size, n_x_cells),
            _extend(self.y_edges, self.y_cell_size, n_y_cells),
            self.z_edges,
        )

    def extended_by_margin(
        self,
        x_margin: pint.Quantity | float,
        y_margin: pint.Quantity | float,
        x_atol: pint.Quantity | float | None = None,
        y_atol: pint.Quantity | float | None = None,
    ) -> CartesianGrid:
        """
        Return a copy extended symmetrically by a margin on each side.

        The cell size is preserved. The margin is rounded down to a whole
        number of cells.

        Parameters
        ----------
        x_margin, y_margin
            Margin added on each side along the X and Y axes. Unitless values
            are interpreted in ``ucc['length']``.
        x_atol, y_atol
            Absolute tolerance on the remainder left when dividing the margin
            by the cell size along the X and Y axes. If ``None``, the remainder
            is ignored. Unitless values are interpreted in ``ucc['length']``.

        Returns
        -------
        CartesianGrid

        Raises
        ------
        ValueError
            If a tolerance is set and the remainder along that axis exceeds it.
        """
        length = ucc.get("length")
        n_cells = []
        for margin, cell_size, atol in [
            (x_margin, self.x_cell_size, x_atol),
            (y_margin, self.y_cell_size, y_atol),
        ]:
            margin = ensure_units(margin, default_units=length)
            if atol is not None:
                atol = ensure_units(atol, default_units=length)
            n_cells.append(_n_cells(margin, cell_size, atol))

        return self.extended_by_cells(*n_cells)

    def cropped(
        self,
        x_extent: pint.Quantity | float,
        y_extent: pint.Quantity | float,
    ) -> CartesianGrid:
        """
        Return a copy cropped symmetrically to a smaller extent.

        Cells are removed in equal numbers on both sides; the requested extent
        is rounded down to a whole number of cells.

        Parameters
        ----------
        x_extent, y_extent
            Target extent along the X and Y axes. Unitless values are
            interpreted in ``ucc['length']``.

        Returns
        -------
        CartesianGrid

        Raises
        ------
        ValueError
            If a target extent exceeds the grid extent, or if the number of
            cells to remove along an axis is odd.
        """
        edges = []
        for axis, extent in [("x", x_extent), ("y", y_extent)]:
            extent = ensure_units(extent, default_units=ucc.get("length"))
            current_extent = getattr(self, f"{axis}_extent")
            if extent > current_extent:
                raise ValueError(
                    f"{axis}_extent ({extent}) exceeds the grid extent "
                    f"({current_extent})"
                )
            n_cells = _n_cells(extent, getattr(self, f"{axis}_cell_size"), None)
            n_drop, remainder = divmod(getattr(self, f"n_{axis}_cells") - n_cells, 2)
            if remainder != 0:
                raise ValueError(
                    "cannot crop symmetrically: the number of cells to remove "
                    f"along {axis.upper()} is odd"
                )
            axis_edges = getattr(self, f"{axis}_edges")
            edges.append(axis_edges[n_drop : len(axis_edges) - n_drop])

        return self.__class__(*edges, self.z_edges)

    def centered(self) -> CartesianGrid:
        """
        Return a copy translated horizontally so that it is centred at the
        origin.

        Returns
        -------
        CartesianGrid
        """
        return self.__class__(
            self._centered_edges(self.x_extent, self.n_x_cells),
            self._centered_edges(self.y_extent, self.n_y_cells),
            self.z_edges,
        )

    def single_column(self) -> CartesianGrid:
        """
        Return a copy with a single horizontal cell spanning the grid extent.

        Returns
        -------
        CartesianGrid
        """
        return self.with_cell_counts(n_x_cells=1, n_y_cells=1)


@frozen(eq=False, init=False)
class SphericalGrid(SpatialGrid):
    """
    Spherical grid [``spherical``].

    Grid for spherical-shell geometries, in a geocentric spherical frame
    (X: azimuth [angle], Y: colatitude [angle], Z: altitude [length]).
    Unitless edges are interpreted in ``ucc['angle']`` along X and Y and in
    ``ucc['length']`` along Z.

    The constructor takes cell edges along each axis, or their aliases
    (exclusive with the generic arguments): ``azimuths`` (X), ``colatitudes``
    (Y) and ``levels`` (Z). Cells along X are called *sectors* and cells
    along Y are called *bands*.

    The azimuth range must be at most 360° wide; see :meth:`kernel_azimuths`
    for the ranges accepted by the kernel.
    """

    _AXIS_UNITS = ("angle", "angle", "length")

    def __init__(
        self,
        x_edges: pint.Quantity | xr.DataArray | None = None,
        y_edges: pint.Quantity | xr.DataArray | None = None,
        z_edges: pint.Quantity | xr.DataArray | None = None,
        *,
        azimuths: pint.Quantity | xr.DataArray | None = None,
        colatitudes: pint.Quantity | xr.DataArray | None = None,
        levels: pint.Quantity | xr.DataArray | None = None,
    ):
        self._init_edges(
            _either(x_edges, azimuths, "x_edges", "azimuths"),
            _either(y_edges, colatitudes, "y_edges", "colatitudes"),
            _either(z_edges, levels, "z_edges", "levels"),
        )
        self.kernel_azimuths()  # Raises if the azimuth range is unsupported

    @classmethod
    def from_levels(cls, levels: pint.Quantity | np.typing.ArrayLike) -> SphericalGrid:
        """
        Build a single-column grid from altitude levels.

        The single cell spans the whole sphere (azimuths 0° to 360°,
        colatitudes 0° to 180°).

        Parameters
        ----------
        levels
            Altitude levels (Z edges). Unitless values are interpreted in
            ``ucc['length']``.

        Returns
        -------
        SphericalGrid
        """
        return cls(
            azimuths=np.asarray([0.0, 360.0]) * ureg.degree,
            colatitudes=np.asarray([0.0, 180.0]) * ureg.degree,
            levels=levels,
        )

    # --------------------------------------------------------------------------
    #                              X/Y-axis aliases
    # --------------------------------------------------------------------------

    @property
    def azimuths(self) -> pint.Quantity:
        """Azimuths of sector edges (alias of :attr:`x_edges`)."""
        return self.x_edges

    @property
    def sectors(self) -> pint.Quantity:
        """Azimuths of sector centres (alias of :attr:`x_centers`)."""
        return self.x_centers

    @property
    def sector_width(self) -> pint.Quantity:
        """Uniform sector width (alias of :attr:`x_cell_size`)."""
        return self.x_cell_size

    @property
    def colatitudes(self) -> pint.Quantity:
        """Colatitudes of band edges (alias of :attr:`y_edges`)."""
        return self.y_edges

    @property
    def bands(self) -> pint.Quantity:
        """Colatitudes of band centres (alias of :attr:`y_centers`)."""
        return self.y_centers

    @property
    def band_width(self) -> pint.Quantity:
        """Uniform band width (alias of :attr:`y_cell_size`)."""
        return self.y_cell_size

    # --------------------------------------------------------------------------
    #                                  Misc
    # --------------------------------------------------------------------------

    def kernel_azimuths(self) -> tuple[float, float, int]:
        """
        Map the azimuth range to the kernel's [-π, π] convention.

        ``sphericalcoordsvolume`` maps the azimuth
        :math:`\\varphi = \\operatorname{atan2}(y, x) \\in [-\\pi, \\pi]` to the
        volume's Z coordinate. The grid's azimuth range is placed in that
        interval as follows:

        * if it fits after a shift by a multiple of 2π, the edges are shifted
          and the data order is unchanged;
        * if it covers the full circle and the ±π cut falls on a cell edge,
          the edges become [-π, π] and the data must be rolled along the
          azimuth axis so that the first kernel cell is the one starting at
          the cut;
        * a single cell covering the full circle maps to [-π, π].

        Returns
        -------
        phi_min, phi_max : float
            Kernel-side azimuth bounds in radians.
        n_roll : int
            Number of cells by which volume data must be rolled backwards
            along the azimuth axis (``np.roll(data, -n_roll, axis=0)``).

        Raises
        ------
        ValueError
            If the range is wider than 2π, or crosses the ±π cut without being a
            full circle with the cut on a cell edge.
        """
        a0, a1 = self.azimuths[[0, -1]].m_as(ureg.rad)
        width = a1 - a0
        two_pi = 2.0 * np.pi
        full = bool(np.isclose(width, two_pi))

        if width <= 0.0 or (width > two_pi and not full):
            raise ValueError(
                "azimuth range must be increasing and at most 2π wide, got "
                f"{self.azimuths[[0, -1]]:~P}"
            )

        # Shift the range start into [-π, π)
        phi_min = a0 - two_pi * np.floor((a0 + np.pi) / two_pi)
        phi_max = phi_min + width
        if phi_max <= np.pi or np.isclose(phi_max, np.pi):
            return float(phi_min), float(phi_max), 0

        if full:
            if self.n_x_cells == 1:
                return -np.pi, np.pi, 0
            n_roll = (np.pi - phi_min) / self.sector_width.m_as(ureg.rad)
            if np.isclose(n_roll, np.round(n_roll)):
                return -np.pi, np.pi, int(np.round(n_roll)) % self.n_x_cells

        raise ValueError(
            f"azimuth range {self.azimuths[[0, -1]]:~P} crosses the ±π cut of "
            "the kernel's [-π, π] azimuth convention; only full-circle ranges "
            "with the cut on a cell edge may cross it"
        )

    @property
    def to_world(self) -> mi.ScalarTransform4f:
        """
        Transform from the unit cube to the part of ``sphericalcoordsvolume``'s
        normalized coordinate space covered by the grid.

        The kernel's volume coordinates are (radius, colatitude, azimuth),
        each normalized to [0, 1]. The grid always covers the full radius
        range; the colatitude and azimuth ranges are those of the grid edges
        (see :meth:`kernel_azimuths`).
        """
        phi_min, phi_max, _ = self.kernel_azimuths()
        theta_min = self.colatitudes.min().m_as(ureg.rad)
        theta_max = self.colatitudes.max().m_as(ureg.rad)

        # sphericalcoordsvolume normalized coordinates:
        #   y = theta / pi           (colatitude)
        #   z = phi / (2 * pi) + 0.5 (azimuth)
        return map_unit_cube(
            xmin=0.0,
            xmax=1.0,  # Radius: always spans [0, 1]
            ymin=theta_min / np.pi,
            ymax=theta_max / np.pi,
            zmin=phi_min / (2 * np.pi) + 0.5,
            zmax=phi_max / (2 * np.pi) + 0.5,
        )

    def __repr__(self) -> str:
        return (
            "<SphericalGrid "
            f"azimuth={self.azimuths[[0, -1]]:~P} ({self.n_x_cells} sectors) "
            f"colatitude={self.colatitudes[[0, -1]]:~P} ({self.n_y_cells} bands) "
            f"z={self.levels[[0, -1]]:~P} ({self.n_layers} layers)"
            ">"
        )
