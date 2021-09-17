__all__ = [
    "detect_axes",
    "get_axes_from_facet_grid",
    "make_ticks",
    "remove_xyticks",
    "remove_xylabels",
]

import typing as t

import matplotlib.pyplot as _plt
import numpy as np
import xarray.plot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray.plot import FacetGrid

# -- Utility functions ---------------------------------------------------------


def detect_axes(from_=None):
    """
    Try and extract a :class:`~matplotlib.axes.Axes` list from a data structure.

    Parameters
    ----------
    from_ : matplotlib figure or axes or list of axes or :class:`xarray.plot.FacetGrid`, optional
        Data structure to get an :class:`~matplotlib.axes.Axes` list from.
        If unset, :func:`matplotlib.pyplot.gca()` is used.

    Returns
    -------
    list of matplotlib axes
        Extracted list of :class:`~matplotlib.axes.Axes`.

    Raises
    ------
    TypeError
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


def get_axes_from_facet_grid(
    facet_grid: xarray.plot.FacetGrid, exclude: str = None
) -> t.List[Axes]:
    """
    Extract a flat list of :class:`~matplotlib.axes.Axes` from a
    :class:`~xarray.plot.FacetGrid`.

    Parameters
    ----------
    facet_grid : :class:`xarray.plot.FacetGrid`
        Object to extract a list of :class:`~matplotlib.axes.Axes`.

    exclude : str, optional
        Exclude selected :class:`~matplotlib.axes.Axes` from
        the returned list. Supported values:

        * ``"lower_left"``: exclude lower-left corner.

    Returns
    -------
    list of matplotlib axes
        List of extracted :class:`~matplotlib.axes.Axes` objects.
    """
    index = np.full((facet_grid.axes.size,), True, dtype=bool)
    n_rows, n_cols = facet_grid.axes.shape

    # This filtering system can be extended in the future
    if exclude == "lower_left":
        index[(n_rows - 1) * n_cols] = False

    return list(facet_grid.axes.flatten()[index])


def remove_xylabels(from_=None) -> None:
    """
    Remove x and y axis labels from ``from_``
    (processed by :func:`detect_axes`).

    Parameters
    ----------
    from_ : matplotlib figure or axes or list of axes or :class:`xarray.plot.FacetGrid`, optional
        Data structure to get an :class:`~matplotlib.axes.Axes` list from.
        If unset, :func:`matplotlib.pyplot.gca()` is used.

    See Also
    --------
    :func:`detect_axes`
    """
    for ax in detect_axes(from_):
        ax.set_xlabel("")
        ax.set_ylabel("")


def remove_xyticks(from_=None) -> None:
    """
    Remove x and y axis tick labels from ``from_``
    (processed by :func:`detect_axes`).

    Parameters
    ----------
    from_ : matplotlib figure or axes or list of axes or :class:`xarray.plot.FacetGrid`, optional
        Data structure to get an :class:`~matplotlib.axes.Axes` list from.
        If unset, :func:`matplotlib.pyplot.gca()` is used.

    See Also
    --------
    :func:`detect_axes`
    """
    for ax in detect_axes(from_):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def make_ticks(num_ticks: int, limits: t.Sequence[float]):
    """
    Generate ticks and their respective tickmarks.

    Parameters
    ----------
    num_ticks : int
        Number of ticks to generate, including the limits
        of the given range

    limits : pair of float
        List of two values, limiting the ticks inclusive

    Returns
    ------
    steps : list of float
        Values for the ticks.

    marks : list of str
        Tick values converted to degrees as string tickmarks.
    """

    step_width = float(limits[1] - limits[0]) / (num_ticks - 1)

    steps = [limits[0] + step_width * i for i in range(num_ticks)]
    marks = [f"{i / np.pi * 180}°" for i in steps]

    return steps, marks
