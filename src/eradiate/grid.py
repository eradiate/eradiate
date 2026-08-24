from __future__ import annotations

from abc import ABC, abstractmethod

import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from eradiate.kernel.transform import map_unit_cube

from . import _repr_diagrams
from .attrs import documented, frozen
from .units import unit_context_config as ucc
from .units import unit_context_kernel as uck
from .units import unit_registry as ureg


@frozen(eq=False, init=False)
class GridCoords(ABC):
    """
    Abstract coordinate grid for an atmosphere, expressed in 3 dimensions (X, Y and Z).

    Eradiate's coordinate systems are made to conveniently locate objects
    in a 3D scene. It does not map to an actual coordinate system used in
    Earth Observation or to a specific projection explicitly.

    Two coordinate systems are used internally, one for spherical shell
    atmospheric geometries and another for plane parallel geometries.

    Specializations of this class may implement their own logic to map their
    coordinate system to one of the internal grid coordinates of Eradiate:

    * Plane parallel: a local Cartesian frame
      (X: local position [L], Y: local position [L], Z: altitude [L])
    * Spherical shell: a geocentric spherical frame
      (X: azimuth [dimensionless], Y: colatitude [dimensionless], Z: radius [L])

    Notes
    -----
    * Instances are immutable.
    * Instances are hashable by ID. This is required to allow for using them as
      an argument of an LRU-cached function.
    * This class is used as the argument of the ``eval()`` family of methods.
    """

    levels: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            on_setattr=None,
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node altitudes.\n\nUnit-enabled field (default: ``ucc['length']``).",
    )

    _layers: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _layer_height: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    @_layer_height.validator
    def _layer_height_validator(self, attribute, value):
        if not np.isscalar(value.magnitude):
            raise ValueError("layer height must be a scalar")

    _total_height: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    @staticmethod
    def make_onedim_from_levels(levels: pint.Quantity):
        """
        Build a plane parallel 1D GridCoords using provided ``levels`` to
        define the vertical extent and the horizontal extent comes from
        Eradiate's default atmosphere width
        """

        ensure_units(levels, default_units=ucc.get("length"))

        default = GridCoords.make_default()

        return PlaneParallelGridCoords(
            levels=levels,
            edges_x=default.edges_x,
            edges_y=default.edges_y,
        )

    @staticmethod
    def make_onedim_arange(
        top: pint.Quantity,
        bottom: pint.Quantity,
        step: pint.Quantity,
        width: pint.Quantity,
    ):
        """
        Build a plane parallel 1D GridCoords using ``width`` for horizontal
        extents and the provided ``top`` and ``bottom`` altitude for the vertical coordinate.

        Parameters
        ----------
        top
            Top of the atmosphere altitude
        bottom
            Ground altitude
        step
            Regular layer height
        width
            Grid width

        Returns
        -------
        PlaneParallelGridCoords
            Constructed coords grid
        """

        ensure_units(top, default_units=ucc.get("length"))
        ensure_units(bottom, default_units=ucc.get("length"))
        ensure_units(width, default_units=ucc.get("length"))

        levels = ureg.convert(
            np.arange(
                bottom.m_as(ureg.m), (top + step * 0.1).m_as(ureg.m), step.m_as(ureg.m)
            ),
            ureg.m,
            ucc.get("length"),
        )

        return PlaneParallelGridCoords.from_extent_and_resolution(
            levels=levels,
            extent_x=width,
            extent_y=width,
            n_cells_x=1,
            n_cells_y=1,
        )

    @staticmethod
    def make_default():
        """
        Build the default plane parallel 1D GridCoords for Eradiate
        atmospheres.

        Bottom altitude is set to zero and top at 120 km. Each layer is 100m high.
        The horizontal extent is 1e6 km.
        """

        bottom = 0.0 * ureg.km
        top = 120.0 * ureg.km
        step = 100.0 * ureg.m
        width = 1e6 * ureg.km

        return GridCoords.make_onedim_arange(top, bottom, step, width)

    @classmethod
    def convert(cls, value: t.Any) -> t.Any:
        """
        Convert dictionary type of parameters to a GridCoords.

        Returns the default plane parallel ``GridCoords`` if value
        is ``"plane_parallel_coords"``. If value is a ``pint.Quantity``,
        attempts to construct a 1D atmosphere with the layer altitudes
        contained in the quantity. If value is a ``np.ndarray``, interpret it
        as an array of layer altitudes in meters.

        If value is a dict, expects a ``"type"`` key with the value
        ``"plane_parallel_coords"`` or ``"spherical_shell_geometry"``.
        Other keys are passed to the relevant subclass constructor.

        Parameters
        ----------
        value
            ``str``, ``dict`` or ``pint.Quantity`` the parameter specification

        Returns
        -------
        GridCoords
            Constructed grid
        """
        if isinstance(value, str):
            if value == "plane_parallel_coords":
                return cls.make_default()
            raise ValueError(f"No default factory for the type id {type}")
        elif isinstance(value, dict):
            value = value.copy()
            value_t = value.pop("type")
            if value_t == "plane_parallel_coords":
                return PlaneParallelGridCoords(**value)
            elif value_t == "spherical_shell_coords":
                return SphericalShellGridCoords(**value)
        elif isinstance(value, pint.Quantity):
            return cls.make_onedim_from_levels(value)
        elif isinstance(value, np.ndarray):
            return cls.make_onedim_from_levels(value * ureg.m)

        return value

    @property
    def layers(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of altitudes of layer centres.
        """
        return self._layers

    @property
    def layer_height(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Layer height.
        """
        return self._layer_height

    @property
    def total_height(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Total height covered by the altitude grid.
        """
        return self._total_height

    @property
    def n_levels(self) -> int:
        """
        Returns
        -------
        int
            Number of levels (Z edges).
        """
        return len(self.levels)

    @property
    def n_layers(self) -> int:
        """
        Returns
        -------
        int
            Number of layers (Z cells).
        """
        return len(self.layers)

    @property
    @abstractmethod
    def cells_x(self) -> pint.Quantity:
        raise NotImplementedError()

    @property
    @abstractmethod
    def cells_y(self) -> pint.Quantity:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_cells_x(self) -> int:
        """Number of cells in the X direction."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_cells_y(self) -> int:
        """Number of cells in the Y direction."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_edges_x(self) -> int:
        """Number of edges in the X direction."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_edges_y(self) -> int:
        """Number of edges in the Y direction."""
        raise NotImplementedError()

    @property
    def n_cells_z(self) -> int:
        """Number of cells in the Z direction (equal to number of layers)."""
        return self.n_layers

    @property
    def n_edges_z(self) -> int:
        """Number of edges in the Z direction (equal to number of levels)."""
        return self.n_levels

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid dimensions sizes, or numpy shape. (X, Y, Z) order."""
        return (self.n_cells_x, self.n_cells_y, self.n_cells_z)

    @property
    def onedim(self) -> bool:
        """Is this grid suitable for expressing properties of a 1D atmosphere."""
        return self.n_cells_x == 1 and self.n_cells_y == 1

    @property
    @abstractmethod
    def to_world(self) -> mi.ScalarTransform4f:
        """Grid to_world"""

    def _repr_html_(self) -> str | None:
        return _repr_diagrams.grid_repr_html(self)


@frozen(eq=False, init=False)
class PlaneParallelGridCoords(GridCoords):
    """
    Plane parallel grid coordinates [``plane_parallel_coords``].

    Implements the internal coordinate system for Eradiate plane parallel
    geometries, using a local Cartesian reference frame.
    (X: local position [L], Y: local position [L], Z: altitude [L])

    The grid is regular: cells have uniform width in X and uniform length in Y,
    but these two spacings are independent.
    """

    edges_x: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            on_setattr=None,
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node positions in the X direction.\n\nUnit-enabled field (default: ``ucc['length']``).",
    )

    edges_y: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            on_setattr=None,
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node positions in the Y direction.\n\nUnit-enabled field (default: ``ucc['length']``).",
    )

    _centers_x: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _centers_y: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _cell_width: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _cell_length: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _total_width: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    _total_length: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,
    )

    def __rich_repr__(self):
        yield "levels", self.levels
        yield "layers", self.layers
        yield "layer_height", self.layer_height
        yield "total_height", self.total_height

        yield "edges_x", self.edges_x
        yield "centers_x", self.centers_x
        yield "cell_width", self.cell_width
        yield "total_width", self.total_width

        yield "edges_y", self.edges_y
        yield "centers_y", self.centers_y
        yield "cell_length", self.cell_length
        yield "total_length", self._total_length

        yield "shape", self.shape

    @_cell_width.validator
    @_cell_length.validator
    def _cell_extent_validator(self, attribute, value):
        if not np.isscalar(value.magnitude):
            raise ValueError("cell extent must be a scalar")

    def __init__(
        self,
        levels: pint.Quantity,
        edges_x: pint.Quantity,
        edges_y: pint.Quantity,
    ):
        levels = ensure_units(levels, default_units=ucc.get("length"))
        layer_height = np.diff(levels)
        if not np.allclose(layer_height, layer_height[0]):
            raise ValueError("levels must be regularly spaced")
        layers = levels[:-1] + 0.5 * layer_height

        edges_x = ensure_units(edges_x, default_units=ucc.get("length"))
        edges_y = ensure_units(edges_y, default_units=ucc.get("length"))

        def _centers_and_step(edges, axis):
            step = np.diff(edges)
            if not np.allclose(step, step[0]):
                raise ValueError(f"edges_{axis} must be regularly spaced")
            return edges[:-1] + 0.5 * step, step[0]

        centers_x, cell_width = _centers_and_step(edges_x, "x")
        centers_y, cell_length = _centers_and_step(edges_y, "y")

        self.__attrs_init__(
            levels=levels,
            layers=layers,
            layer_height=layer_height[0],
            total_height=levels[-1] - levels[0],
            edges_x=edges_x,
            edges_y=edges_y,
            centers_x=centers_x,
            centers_y=centers_y,
            cell_width=cell_width,
            cell_length=cell_length,
            total_width=edges_x[-1] - edges_x[0],
            total_length=edges_y[-1] - edges_y[0],
        )

    @property
    def to_world(self) -> mi.ScalarTransform4f:
        """Grid to_world in cartesian coordinates"""
        unit = uck.get("length")
        return map_unit_cube(
            self.edges_x.min().m_as(unit),
            self.edges_x.max().m_as(unit),
            self.edges_y.min().m_as(unit),
            self.edges_y.max().m_as(unit),
            self.levels.min().m_as(unit),
            self.levels.max().m_as(unit),
        )

    @property
    def centers_x(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of cell centre positions in the X direction.
        """
        return self._centers_x

    @property
    def centers_y(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of cell centre positions in the Y direction.
        """
        return self._centers_y

    @property
    def cell_width(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Uniform cell width in the X direction.
        """
        return self._cell_width

    @property
    def cell_length(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Uniform cell length in the Y direction.
        """
        return self._cell_length

    @property
    def total_width(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Total extent of the grid in the X direction.
        """
        return self._total_width

    @property
    def total_length(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Total extent of the grid in the Y direction.
        """
        return self._total_length

    @property
    def n_cells_x(self) -> int:
        """Number of cells in the X direction."""
        return len(self._centers_x)

    @property
    def n_cells_y(self) -> int:
        """Number of cells in the Y direction."""
        return len(self._centers_y)

    @property
    def n_edges_x(self) -> int:
        """Number of edges in the X direction."""
        return len(self.edges_x)

    @property
    def n_edges_y(self) -> int:
        """Number of edges in the Y direction."""
        return len(self.edges_y)

    def __repr__(self) -> str:
        return (
            f"PlaneParallelGridCoords(\n"
            f"  x: [{self.edges_x[0]:~P}, {self.edges_x[-1]:~P}], {self.n_cells_x} cells\n"
            f"  y: [{self.edges_y[0]:~P}, {self.edges_y[-1]:~P}], {self.n_cells_y} cells\n"
            f"  z: [{self.levels[0]:~P}, {self.levels[-1]:~P}], {self.n_layers} layers\n"
            f")"
        )

    @staticmethod
    def _make_centered_edges(extent: pint.Quantity, n_cells: int) -> pint.Quantity:
        """Return regularly spaced edges centred at zero."""
        half = extent / 2.0
        return np.linspace(-half, half, n_cells + 1)

    @staticmethod
    def _cells_from_extent_and_size(
        extent: pint.Quantity,
        cell_size: pint.Quantity,
        atol: pint.Quantity | None,
    ) -> int:
        """
        Derive the number of cells fitting in *extent* given *cell_size*.

        Parameters
        ----------
        extent
            Total extent of the domain.
        cell_size
            Desired cell size.
        atol
            Absolute tolerance on the remainder. If ``None``, the remainder is
            silently ignored and the floor count is returned. Otherwise a
            ``ValueError`` is raised when the remainder exceeds *atol*.

        Returns
        -------
        int
            Number of cells.
        """
        ratio = extent / cell_size
        n = int(np.floor(ratio.to("").magnitude))
        remainder = (ratio.to("").magnitude - n) * cell_size
        if atol is not None and remainder > atol:
            raise ValueError(
                f"extent {extent} is not a multiple of cell size {cell_size} "
                f"within tolerance {atol} (remainder: {remainder})"
            )
        return n

    @classmethod
    def from_extent_and_resolution(
        cls,
        levels: pint.Quantity,
        extent_x: pint.Quantity,
        extent_y: pint.Quantity,
        n_cells_x: int,
        n_cells_y: int,
    ) -> PlaneParallelGridCoords:
        """
        Construct from domain extents and cell counts, centred at the origin.

        Parameters
        ----------
        levels
            Altitude levels (Z edges).
        extent_x
            Total extent of the domain in the X direction.
        extent_y
            Total extent of the domain in the Y direction.
        n_cells_x
            Number of cells in the X direction.
        n_cells_y
            Number of cells in the Y direction.

        Returns
        -------
        PlaneParallelGridCoords
        """
        extent_x = ensure_units(extent_x, default_units=ucc.get("length"))
        extent_y = ensure_units(extent_y, default_units=ucc.get("length"))

        edges_x = cls._make_centered_edges(extent_x, n_cells_x)
        edges_y = cls._make_centered_edges(extent_y, n_cells_y)

        return cls(levels=levels, edges_x=edges_x, edges_y=edges_y)

    @classmethod
    def from_extent_and_cell_size(
        cls,
        levels: pint.Quantity,
        extent_x: pint.Quantity,
        extent_y: pint.Quantity,
        cell_width: pint.Quantity,
        cell_length: pint.Quantity,
        atol: pint.Quantity | None = None,
    ) -> PlaneParallelGridCoords:
        """
        Construct from domain extents and cell sizes, centred at the origin.

        The number of cells is derived from the extent and cell size. If the
        extent is not an exact multiple of the cell size, the behaviour depends
        on *atol*: if ``None``, the largest fitting number of cells is used;
        otherwise a ``ValueError`` is raised when the remainder exceeds *atol*.

        Parameters
        ----------
        levels
            Altitude levels (Z edges).
        extent_x
            Total extent of the domain in the X direction.
        extent_y
            Total extent of the domain in the Y direction.
        cell_width
            Desired cell width in the X direction.
        cell_length
            Desired cell length in the Y direction.
        atol
            Absolute tolerance on the remainder extent. Unit-enabled.

        Returns
        -------
        PlaneParallelGridCoords
        """
        extent_x = ensure_units(extent_x, default_units=ucc.get("length"))
        extent_y = ensure_units(extent_y, default_units=ucc.get("length"))
        cell_width = ensure_units(cell_width, default_units=ucc.get("length"))
        cell_length = ensure_units(cell_length, default_units=ucc.get("length"))
        if atol is not None:
            atol = ensure_units(atol, default_units=ucc.get("length"))

        n_cells_x = cls._cells_from_extent_and_size(extent_x, cell_width, atol)
        n_cells_y = cls._cells_from_extent_and_size(extent_y, cell_length, atol)

        edges_x = cls._make_centered_edges(extent_x, n_cells_x)
        edges_y = cls._make_centered_edges(extent_y, n_cells_y)

        return cls(levels=levels, edges_x=edges_x, edges_y=edges_y)

    @classmethod
    def from_cell_size_and_count(
        cls,
        levels: pint.Quantity,
        cell_width: pint.Quantity,
        cell_length: pint.Quantity,
        n_cells_x: int,
        n_cells_y: int,
    ) -> PlaneParallelGridCoords:
        """
        Construct from cell sizes and counts, centred at the origin.

        The domain extent is derived as ``cell_size * n_cells``.

        Parameters
        ----------
        levels
            Altitude levels (Z edges).
        cell_width
            Cell width in the X direction.
        cell_length
            Cell length in the Y direction.
        n_cells_x
            Number of cells in the X direction.
        n_cells_y
            Number of cells in the Y direction.

        Returns
        -------
        PlaneParallelGridCoords
        """
        cell_width = ensure_units(cell_width, default_units=ucc.get("length"))
        cell_length = ensure_units(cell_length, default_units=ucc.get("length"))

        extent_x = cell_width * n_cells_x
        extent_y = cell_length * n_cells_y

        edges_x = cls._make_centered_edges(extent_x, n_cells_x)
        edges_y = cls._make_centered_edges(extent_y, n_cells_y)

        return cls(levels=levels, edges_x=edges_x, edges_y=edges_y)

    def extended_by_cells(
        self,
        n_x: int,
        n_y: int,
    ) -> PlaneParallelGridCoords:
        """
        Extend the grid symmetrically by appending cells on each side.

        The cell size is preserved. ``n_x`` cells are added on each side in X,
        and ``n_y`` cells on each side in Y, so the total cell count grows by
        ``2 * n_x`` and ``2 * n_y`` respectively.

        Parameters
        ----------
        n_x
            Number of cells to add on each side in the X direction.
        n_y
            Number of cells to add on each side in the Y direction.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with extended edges.
        """

        def _extend(edges, step, n):
            prepend = edges[0] - step * np.arange(n, 0, -1)
            append = edges[-1] + step * np.arange(1, n + 1)
            return np.concatenate([prepend, edges, append])

        edges_x = _extend(self.edges_x, self.cell_width, n_x)
        edges_y = _extend(self.edges_y, self.cell_length, n_y)

        return self.__class__(levels=self.levels, edges_x=edges_x, edges_y=edges_y)

    def extended_by_extent(
        self,
        extent_x: pint.Quantity,
        extent_y: pint.Quantity,
        atol: pint.Quantity | None = None,
    ) -> PlaneParallelGridCoords:
        """
        Extend the grid symmetrically by a physical margin on each side.

        The cell size is preserved. The number of cells added is derived from
        the margin and cell size. If the margin is not an exact multiple of the
        cell size, the behaviour depends on *atol*.

        Parameters
        ----------
        extent_x
            Physical margin to add on each side in the X direction.
        extent_y
            Physical margin to add on each side in the Y direction.
        atol
            Absolute tolerance on the remainder extent.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with extended edges.
        """
        extent_x = ensure_units(extent_x, default_units=ucc.get("length"))
        extent_y = ensure_units(extent_y, default_units=ucc.get("length"))
        if atol is not None:
            atol = ensure_units(atol, default_units=ucc.get("length"))

        n_x = self._cells_from_extent_and_size(extent_x, self.cell_width, atol)
        n_y = self._cells_from_extent_and_size(extent_y, self.cell_length, atol)

        return self.extended_by_cells(n_x, n_y)

    def resampled(
        self,
        n_cells_x: int,
        n_cells_y: int,
    ) -> PlaneParallelGridCoords:
        """
        Resample the grid to a new resolution, preserving the extent.

        Parameters
        ----------
        n_cells_x
            New number of cells in the X direction.
        n_cells_y
            New number of cells in the Y direction.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with the same extent but different resolution.
        """
        edges_x = np.linspace(self.edges_x[0], self.edges_x[-1], n_cells_x + 1)
        edges_y = np.linspace(self.edges_y[0], self.edges_y[-1], n_cells_y + 1)

        return self.__class__(levels=self.levels, edges_x=edges_x, edges_y=edges_y)

    def resampled_to_cell_size(
        self,
        cell_width: pint.Quantity,
        cell_length: pint.Quantity,
        atol: pint.Quantity | None = None,
    ) -> PlaneParallelGridCoords:
        """
        Resample the grid to a new cell size, preserving the extent.

        Parameters
        ----------
        cell_width
            New cell width in the X direction.
        cell_length
            New cell length in the Y direction.
        atol
            Absolute tolerance on the remainder extent.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with the same extent but different cell size.
        """
        cell_width = ensure_units(cell_width, default_units=ucc.get("length"))
        cell_length = ensure_units(cell_length, default_units=ucc.get("length"))
        if atol is not None:
            atol = ensure_units(atol, default_units=ucc.get("length"))

        n_cells_x = self._cells_from_extent_and_size(self.total_width, cell_width, atol)
        n_cells_y = self._cells_from_extent_and_size(
            self.total_length, cell_length, atol
        )

        return self.resampled(n_cells_x, n_cells_y)

    def cropped(
        self,
        extent_x: pint.Quantity,
        extent_y: pint.Quantity,
    ) -> PlaneParallelGridCoords:
        """
        Crop the grid symmetrically to a smaller extent.

        Snaps to the nearest existing edge inward. Raises if the requested
        extent exceeds the current domain.

        Parameters
        ----------
        extent_x
            Target extent in the X direction.
        extent_y
            Target extent in the Y direction.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with reduced extent.

        Raises
        ------
        ValueError
            If the requested extent exceeds the current domain in either
            direction.
        """
        extent_x = ensure_units(extent_x, default_units=ucc.get("length"))
        extent_y = ensure_units(extent_y, default_units=ucc.get("length"))

        if extent_x > self.total_width:
            raise ValueError(
                f"requested extent_x {extent_x} exceeds current domain width {self.total_width}"
            )
        if extent_y > self.total_length:
            raise ValueError(
                f"requested extent_y {extent_y} exceeds current domain length {self.total_length}"
            )

        def _crop_edges(edges, n_drop):
            return edges[n_drop : len(edges) - n_drop]

        def _n_drop(total_cells, target_cells):
            drop, remainder = divmod(total_cells - target_cells, 2)
            if remainder != 0:
                raise ValueError(
                    f"cannot crop symmetrically: difference in cell count "
                    f"({total_cells - target_cells}) is odd"
                )
            return drop

        n_cells_x = self._cells_from_extent_and_size(
            extent_x, self.cell_width, atol=None
        )
        n_cells_y = self._cells_from_extent_and_size(
            extent_y, self.cell_length, atol=None
        )

        drop_x = _n_drop(self.n_cells_x, n_cells_x)
        drop_y = _n_drop(self.n_cells_y, n_cells_y)

        edges_x = _crop_edges(self.edges_x, drop_x)
        edges_y = _crop_edges(self.edges_y, drop_y)

        return self.__class__(levels=self.levels, edges_x=edges_x, edges_y=edges_y)

    def centered(self) -> PlaneParallelGridCoords:
        """
        Return a new grid with the same cell size and count, centred at zero.

        Returns
        -------
        PlaneParallelGridCoords
            New instance centred at the origin.
        """
        edges_x = self._make_centered_edges(self.total_width, self.n_cells_x)
        edges_y = self._make_centered_edges(self.total_length, self.n_cells_y)

        return self.__class__(levels=self.levels, edges_x=edges_x, edges_y=edges_y)

    def single_column(self) -> PlaneParallelGridCoords:
        """
        Collapse the grid to a single horizontal cell.

        Useful when treating a 3D grid as a 1D vertical profile. The single
        cell spans the full extent of the original grid.

        Returns
        -------
        PlaneParallelGridCoords
            New instance with a single cell in X and Y.
        """
        return self.__class__(
            levels=self.levels,
            edges_x=self.edges_x[[0, -1]],
            edges_y=self.edges_y[[0, -1]],
        )

    @property
    def cells_x(self):
        return self.centers_x

    @property
    def cells_y(self):
        return self.centers_y


@frozen(eq=False, init=False)
class SphericalShellGridCoords(GridCoords):
    """
    Spherical shell grid coordinates [``spherical_shell_coords``].

    Implements the internal coordinate system for Eradiate spherical shell
    atmospheric geometries, using a geocentric spherical convention.
    (X: azimuth [dimensionless], Y: colatitude [dimensionless], Z: radius [L])

    Both X and Y coordinates are dimensionless angles (radians in SI).
    The grid is regular in both directions, and independently sized.
    """

    azimuths: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("angle"),
            on_setattr=None,
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node azimuths (X edges).\n\nUnit-enabled field (default: ``ucc['angle']``).",
    )

    colatitudes: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("angle"),
            on_setattr=None,
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node colatitudes (Y edges).\n\nUnit-enabled field (default: ``ucc['angle']``).",
    )

    _sectors: pint.Quantity = pinttr.field(
        units=ucc.deferred("angle"),
        on_setattr=None,
    )

    _bands: pint.Quantity = pinttr.field(
        units=ucc.deferred("angle"),
        on_setattr=None,
    )

    _sector_width: pint.Quantity = pinttr.field(
        units=ucc.deferred("angle"),
        on_setattr=None,
    )

    _band_width: pint.Quantity = pinttr.field(
        units=ucc.deferred("angle"),
        on_setattr=None,
    )

    @_sector_width.validator
    @_band_width.validator
    def _angular_extent_validator(self, attribute, value):
        if not np.isscalar(value.magnitude):
            raise ValueError("angular cell extent must be a scalar")

    def __init__(
        self,
        levels: pint.Quantity,
        azimuths: pint.Quantity,
        colatitudes: pint.Quantity,
    ):
        levels = ensure_units(levels, default_units=ucc.get("length"))
        layer_height = np.diff(levels)
        if not np.allclose(layer_height, layer_height[0]):
            raise ValueError("levels must be regularly spaced")
        layers = levels[:-1] + 0.5 * layer_height

        azimuths = ensure_units(azimuths, default_units=ucc.get("angle"))
        colatitudes = ensure_units(colatitudes, default_units=ucc.get("angle"))

        def _centers_and_step(edges, name):
            step = np.diff(edges)
            if not np.allclose(step, step[0]):
                raise ValueError(f"{name} must be regularly spaced")
            return edges[:-1] + 0.5 * step, step[0]

        sectors, sector_width = _centers_and_step(azimuths, "azimuths")
        bands, band_width = _centers_and_step(colatitudes, "colatitudes")

        self.__attrs_init__(
            levels=levels,
            layers=layers,
            layer_height=layer_height[0],
            total_height=levels[-1] - levels[0],
            azimuths=azimuths,
            colatitudes=colatitudes,
            sectors=sectors,
            bands=bands,
            sector_width=sector_width,
            band_width=band_width,
        )

    @property
    def sectors(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of azimuthal cell centres (X cells).
        """
        return self._sectors

    @property
    def bands(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of colatitudinal cell centres (Y cells).
        """
        return self._bands

    @property
    def sector_width(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Uniform azimuthal cell width.
        """
        return self._sector_width

    @property
    def band_width(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Uniform colatitudinal cell width.
        """
        return self._band_width

    @property
    def n_cells_x(self) -> int:
        """Number of cells in the X (azimuthal) direction."""
        return len(self._sectors)

    @property
    def n_cells_y(self) -> int:
        """Number of cells in the Y (colatitudinal) direction."""
        return len(self._bands)

    @property
    def n_edges_x(self) -> int:
        """Number of edges in the X (azimuthal) direction."""
        return len(self.azimuths)

    @property
    def n_edges_y(self) -> int:
        """Number of edges in the Y (colatitudinal) direction."""
        return len(self.colatitudes)

    def __repr__(self) -> str:
        return (
            f"SphericalShellGridCoords(\n"
            f"  azimuth:    [{self.azimuths[0]:~P}, {self.azimuths[-1]:~P}], {self.n_cells_x} sectors\n"
            f"  colatitude: [{self.colatitudes[0]:~P}, {self.colatitudes[-1]:~P}], {self.n_cells_y} bands\n"
            f"  z:          [{self.levels[0]:~P}, {self.levels[-1]:~P}], {self.n_layers} layers\n"
            f")"
        )

    @property
    def cells_x(self):
        return self.sectors

    @property
    def cells_y(self):
        return self.bands

    @property
    def to_world(self) -> mi.ScalarTransform4f:
        """
        Spherical grid to_world

        This to_world is a portion of the unit cube representing
        the complete extent of geocentric coordinate system.

        This portion corresponds to the extent of the grid colatitudes,
        azimuths and altitude.
        """

        phi_min = self.azimuths.min().m_as(ureg.rad)
        phi_max = self.azimuths.max().m_as(ureg.rad)
        theta_min = self.colatitudes.min().m_as(ureg.rad)
        theta_max = self.colatitudes.max().m_as(ureg.rad)

        # SphericalCoordsVolume outputs:
        #   y = theta / pi        → colatitude normalized to [0, 1]
        #   z = phi / (2*pi) + 0.5 → azimuth normalized to [0, 1]
        return map_unit_cube(
            xmin=0.0,
            xmax=1.0,  # radius: always spans full [0, 1]
            ymin=theta_min / np.pi,
            ymax=theta_max / np.pi,
            zmin=phi_min / (2 * np.pi) + 0.5,
            zmax=phi_max / (2 * np.pi) + 0.5,
        )

    def __rich_repr__(self):
        yield "levels", self.levels
        yield "layers", self.layers
        yield "layer_height", self.layer_height
        yield "total_height", self.total_height

        yield "azimuths", self.azimuths
        yield "sectors", self.sectors
        yield "sector_width", self.sector_width

        yield "colatitudes", self.colatitudes
        yield "bands", self.bands
        yield "band_width", self.band_width

        yield "shape", self.shape
