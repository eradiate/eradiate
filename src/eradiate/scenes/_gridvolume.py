from __future__ import annotations

from functools import partial
from typing import Callable

import mitsuba as mi
import numpy as np

from .geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from ..contexts import KernelContext
from ..kernel import (
    DictParameter,
    KernelSceneParameterFlags,
    SceneParameter,
    SearchSceneParameter,
)


def _prepare_grid(
    eval_grid, ctx, spectral_index, dtype, shape_xyz, extra_kwargs=None, units=None
) -> np.ndarray:
    """
    Broadcasts the return value of the ``eval_grid`` callable onto the
    requested ``shape_xyz``.

    If shape_xyz is ``None``, the eval_grid returns one dimensional
    properties corresponding to a 1D plane parallel atmosphere geometry.

    Otherwise Scalars and 1-D z-column arrays are broadcast to ``shape_xyz``.
    Any other shape mismatch raises a ``ValueError``.

    Performs data type conversion of evaluated grids to ``dtype``. Optionally
    performs units conversion to ``units``.

    ``extra_kwargs`` are passed to eval_grid.

    Returns
    -------
    np.ndarray
        broadcasted evaluated array
    """
    grid = eval_grid(ctx.si if spectral_index else ctx, **extra_kwargs)
    if units:
        grid = grid.m_as(units)
    grid = np.asarray(grid, dtype=dtype)
    assert np.size(grid) > 0
    if shape_xyz is None:
        if np.squeeze(grid).ndim > 1:
            raise ValueError(
                "Evaluation of a non 1D grid for an undefined target shape "
                f"Evaluated grid shape: {grid.shape} is not 1D. Could not "
                "infer a grid volume shape."
            )
        return grid.reshape(1, 1, -1)
    if grid.size == 1:
        return np.broadcast_to(grid, shape_xyz)
    elif np.squeeze(grid).shape == (shape_xyz[2],):
        return np.broadcast_to(grid.reshape(1, 1, -1), shape_xyz)
    if grid.shape != shape_xyz:
        raise ValueError(f"Invalid grid shape, expected {shape_xyz}, got {grid.shape}")
    return grid


def make_volume_grid(
    geometry: SceneGeometry,
    ctx: KernelContext,
    eval_grid=None,
    dtype=None,
    units=None,
    spectral_index=True,
    to_mi=True,
    extra_dim=False,
    extra_kwargs=None,
) -> np.ndarray | mi.VolumeGrid:
    """
    Evaluate a grid function and reshape it to the layout expected by Mitsuba
    for the given geometry.

    Optionally append a trailing dimension and convert to a Mitsuba
    ``VolumeGrid``.

    Raises ``ValueError`` if geometry is None and the grid is not one dimensional.

    For :class:`.PlaneParallelGeometry` the grid is transposed from
    ``(x, y, z)`` to ``(z, y, x)``. For :class:`.SphericalShellGeometry` the
    ``(x, y, z)`` layout is used as-is.

    Returns
    -------
    Union[np.ndarray, mi.VolumeGrid]
        Broadcasted and ordered VolumeGrid
    """
    shape = None
    if geometry is not None:
        shape = geometry.grid.shape
    grid = _prepare_grid(
        eval_grid,
        ctx,
        spectral_index,
        dtype,
        shape,
        extra_kwargs=extra_kwargs,
        units=units,
    )
    if isinstance(geometry, PlaneParallelGeometry):
        grid = grid.T
    if extra_dim:
        grid = grid.reshape(*grid.shape, 1)
    if to_mi:
        return mi.VolumeGrid(grid)
    return grid


class _partial(partial):
    """
    A :class:`functools.partial` subclass with a human-readable ``__repr__``
    and a ``get_bound_instance`` helper that returns the object to which
    ``eval_grid`` is bound.
    """

    def __repr__(self):
        instance = self.get_bound_instance()
        if instance is None:
            cls_name = None
        else:
            cls_name = instance.__class__.__name__
        return f"partial({self.func.__name__}, {cls_name}, extra_parameters={list(self.keywords)})"

    def get_bound_instance(self):
        """Return the instance to which the ``eval_grid`` method is bound."""
        return self.keywords["eval_grid"].__dict__.get("__self__", None)


def _postprocess_template(
    geometry: SceneGeometry,
    partial_factory: Callable,
    filter_type_kw: str,
    wrap_mode_kw: str,
) -> dict:
    """
    Build a Mitsuba kernel dict for a gridvolume plugin, wrapping it in a
    ``sphericalcoordsvolume`` for :class:`.SphericalShellGeometry`.
    """
    gridvolume = {
        "type": "gridvolume",
        "grid": DictParameter(partial_factory),
        "use_grid_bbox": True,
        "filter_type": filter_type_kw,
        "wrap_mode": wrap_mode_kw,
    }
    if geometry is not None:
        to_world = geometry.grid.to_world
        gridvolume["to_world"] = to_world
    if isinstance(geometry, SphericalShellGeometry):
        return {
            "type": "sphericalcoordsvolume",
            "volume": gridvolume,
            "rmin": geometry.atmosphere_volume_rmin,
            "to_world": geometry.atmosphere_volume_to_world,
        }
    return gridvolume


def generate_gridvolume(
    geometry: SceneGeometry | None,
    eval_grid: Callable,
    flag: KernelSceneParameterFlags | None = None,
    search: SearchSceneParameter | None = None,
    filter_type: str | None = None,
    wrap_mode: str | None = None,
    spectral_index: bool = True,
    units=None,
    dtype: type = np.float32,
    extra_kwargs: dict | None = None,
) -> dict | SceneParameter:
    """
    Generate a gridvolume kernel dict or a scene parameter update object.

    When ``flag`` is ``None``, a Mitsuba kernel dict suitable for scene
    construction is returned. When ``flag`` is set, a :class:`.SceneParameter`
    intended for scene parameter updates is returned instead.

    If geometry is None, the 1D plane parallel case is assumed and the Z
    coordinate shape is inferred from the return value of ``eval_grid``.

    Parameters
    ----------
    geometry : .SceneGeometry or NoneType
        Scene geometry driving the grid shape and coordinate layout.

    eval_grid : callable
        A bound method returning the grid data. It must accept a
        :class:`.KernelContext` (or spectral index if ``spectral_index`` is ``True``)
        as its first argument, followed by any ``extra_kwargs``.

        Expected return shape (in ``(x, y, z)`` order):

        * ``()`` or ``(1,)`` — scalar, broadcast to full shape.
        * ``(n_z,)`` — 1-D vertical profile, broadcast along x and y.
        * ``(n_x, n_y, n_z)`` — full 3-D grid.

    flag : object, optional
        Update flag passed to :class:`.SceneParameter`. If ``None`` (default),
        a kernel dict is returned.

    search : .SearchSceneParameter, optional
        Parameter lookup strategy forwarded to :class:`.SceneParameter`.
        Only used when ``flag`` is set.

    filter_type : str, optional
        Mitsuba filter type string. Defaults to ``str(geometry.filter_type)``.

    wrap_mode : str, optional
        Mitsuba wrap mode string. Defaults to ``str(geometry.wrap_mode)``.

    spectral_index : bool, optional, default: False
        If ``True``, pass ``ctx.si`` instead of ``ctx`` to ``eval_grid``.

    units : pint.Unit, optional
    If set, convert grid values to these units before processing.

    dtype : numpy.dtype, optional, default: numpy.float32
        NumPy dtype of the resulting grid buffer.

    extra_kwargs : dict, optional
        Additional keyword arguments forwarded to ``eval_grid``.

    Returns
    -------
    dict
        Mitsuba kernel dict, when ``flag`` is ``None``.

    .SceneParameter
        Scene parameter update object, when ``flag`` is set.

    Examples
    --------
    Generate a kernel dict for a plane-parallel scene:

    .. code-block:: python

        generate_gridvolume(geometry, my_component.eval_sigma_t)

    Generate a scene parameter for spectral updates:

    .. code-block:: python

        generate_gridvolume(
            geometry,
            my_component.eval_sigma_t,
            flag=UpdateFlags.SPECTRAL,
            search=SearchSceneParameter(...),
        )
    """
    pf = _partial(
        make_volume_grid,
        geometry,
        eval_grid=eval_grid,
        dtype=dtype,
        units=units,
        spectral_index=spectral_index,
        extra_dim=flag is not None,
        to_mi=flag is None,
        extra_kwargs=extra_kwargs or {},
    )
    if flag is not None:
        return SceneParameter(pf, flag, search=search)
    if geometry is None:
        filter_type = str(filter_type or "nearest")
        wrap_mode = str(wrap_mode or "clamp")
    else:
        filter_type = str(filter_type or geometry.filter_type)
        wrap_mode = str(wrap_mode or geometry.wrap_mode)

    return _postprocess_template(
        geometry,
        pf,
        filter_type,
        wrap_mode,
    )
