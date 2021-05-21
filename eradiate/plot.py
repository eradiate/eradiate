__all__ = [
    "detect_axes",
    "get_axes_from_facet_grid",
    "make_ticks",
    "remove_xyticks",
    "remove_xylabels",
]

import matplotlib.pyplot as _plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray.plot import FacetGrid

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
