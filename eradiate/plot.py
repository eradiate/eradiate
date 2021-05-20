"""Utility components used to plot data."""

__all__ = [
    "detect_axes",
    "get_axes_from_facet_grid",
    "make_ticks",
    "pcolormesh_polar",
    "remove_xyticks",
    "remove_xylabels",
]

import matplotlib.pyplot as _plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray.plot import FacetGrid

from . import unit_registry as ureg
from ._units import to_quantity

# -- Plotting wrappers ---------------------------------------------------------


def pcolormesh_polar(darray, r=None, theta=None, **kwargs):
    """Create a polar pcolormesh plot. Wraps :func:`xarray.plot.pcolormesh`:
    see its documentation for undocumented keyword arguments.

    .. warning::

       This function might perform write operations on ``darray``: if ``theta``
       if given in degrees (inferred from metadata ``attrs["units"]``), it will
       add to ``darray`` an extra dimension ``theta + "_rad"`` converting
       ``theta`` to radian and use it to generate the plot.

    Parameter ``darray`` (:class:`~xarray.DataArray`):
        Data array to visualise.

    Parameter ``r`` (str or None):
        Radial coordinate. If ``None``, defaults to ``darray.dims[0]``.

    Parameter ``theta`` (str):
        Angular coordinate. If ``None``, defaults to ``darray.dims[1]``.

    Returns → :class:`matplotlib.collections.QuadMesh`:
        Created artist.
    """

    if r is None:
        r = darray.dims[0]
    if theta is None:
        theta = darray.dims[1]

    theta_dims = darray[theta].dims

    try:
        theta_units = ureg.Unit(darray[theta].attrs["units"])
    except KeyError:
        theta_units = ureg.rad

    if theta_units != ureg.rad:
        theta_rad = f"{theta}_rad"
        theta_rad_values = to_quantity(darray[theta]).m_as("rad")
        darray_plot = darray.assign_coords(
            **{theta_rad: (theta_dims, theta_rad_values)}
        )
        darray_plot[theta_rad].attrs = darray[theta].attrs
        darray_plot[theta_rad].attrs["units"] = "rad"
    else:
        theta_rad = theta
        darray_plot = darray

    kwargs["x"] = theta_rad
    kwargs["y"] = r

    subplot_kws = kwargs.get("subplot_kws", None)

    if kwargs.get("ax", None) is None:
        if subplot_kws is None:
            subplot_kws = {}
        subplot_kws["projection"] = "polar"
        kwargs["subplot_kws"] = subplot_kws
    else:
        if subplot_kws is not None:
            raise ValueError("cannot use subplot_kws with existing ax")

    return darray_plot.plot.pcolormesh(**kwargs)


# -- Utility functions ---------------------------------------------------------


def detect_axes(from_=None):
    """Try and extract a :class:`~matplotlib.axes.Axes` list from a data structure.

    Parameter ``from_`` (:class:`~matplotlib.figure.Figure` or :class:`~matplotlib.axes.Axes` or :class:`~xarray.plot.FacetGrid` or list[:class:`~matplotlib.axes.Axes`] or None):
        Data structure to get an :class:`~matplotlib.axes.Axes` list from.
        If ``None``, :func:`matplotlib.pyplot.gca()` is used.

    Returns → list[:class:`~matplotlib.axes.Axes`]:
        Extracted list of :class:`~matplotlib.axes.Axes`.

    Raises → TypeError:
        If ``from_`` is of unsupported type.
    """
    if from_ is None:
        from_ = _plt.gca()

    if isinstance(from_, Figure):
        return from_.axes

    if isinstance(from_, Axes):
        return [from_]

    if isinstance(from_, FacetGrid):
        return list(from_.axes.flatten())

    if isinstance(from_, list):
        if all([isinstance(x, Axes) for x in from_]):
            return from_

    raise TypeError("unsupported type")


def get_axes_from_facet_grid(facet_grid, exclude=None):
    """Extract a flat list of :class:`~matplotlib.axes.Axes` from a
    :class:`~xarray.plot.FacetGrid`.

    Parameter ``facet_grid`` (:class:`~xarray.plot.FacetGrid`):
        Object to extract a list of :class:`~matplotlib.axes.Axes`.

    Parameter ``exclude`` (str or None):
        If not ``None``, exclude selected :class:`~matplotlib.axes.Axes` from
        the returned list. Supported values:

        * ``"lower_left"``: exclude lower-left corner.

    Returns → list[:class:`~matplotlib.axes.Axes`]:
        List of :class:`~xarray.plot.FacetGrid`.
    """
    index = np.full((facet_grid.axes.size,), True, dtype=bool)
    n_rows, n_cols = facet_grid.axes.shape

    # This filtering system can be extended in the future
    if exclude == "lower_left":
        index[(n_rows - 1) * n_cols] = False

    return list(facet_grid.axes.flatten()[index])


def remove_xylabels(from_=None):
    """Remove x and y axis labels from ``from_`` (processed by :func:`detect_axes`)."""
    for ax in detect_axes(from_):
        ax.set_xlabel("")
        ax.set_ylabel("")


def remove_xyticks(from_=None):
    """Remove x and y axis tick labels from ``from_`` (processed by :func:`detect_axes`)."""
    for ax in detect_axes(from_):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def make_ticks(num_ticks, limits):
    """Generates ticks and their respective tickmarks.

    Parameter ``num_ticks`` (int):
        Number of ticks to generate, including the limits
        of the given range

    Parameter ``limits`` (list[float]):
        List of two values, limiting the ticks inclusive

    Returns → list, list:
        - Values for the ticks
        - Tick values converted to degrees as string tickmarks
    """

    step_width = float(limits[1] - limits[0]) / (num_ticks - 1)

    steps = [limits[0] + step_width * i for i in range(num_ticks)]
    marks = [f"{i / np.pi * 180}°" for i in steps]

    return steps, marks
