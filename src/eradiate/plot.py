from __future__ import annotations

import typing as t
import warnings

import matplotlib.pyplot as _plt
import numpy as np
import seaborn as sns
import xarray as xr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from xarray.plot import FacetGrid

from .radprops import CKDAbsorptionDatabase, MonoAbsorptionDatabase


def set_style(rc=None):
    """
    Apply Eradiate Matplotlib style (the Seaborn ``ticks`` style with the
    viridis colormap).

    Parameters
    ----------
    rc : dict, optional
        A Matplotlib rc parameter dictionary to be applied in addition to the
        Eradiate style.
    """
    try:
        import seaborn
    except ModuleNotFoundError:
        warnings.warn(
            "To use Eradiate's Matplotlib style, you must install Seaborn.\n"
            "See instructions on https://seaborn.pydata.org/installing.html."
        )
        raise

    if rc is None:
        rc = {}

    seaborn.set_theme(style="ticks", rc={"image.cmap": "viridis", **rc})


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


def get_axes_from_facet_grid(facet_grid: FacetGrid, exclude: str = None) -> list[Axes]:
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
    Remove x- and y-axis labels from ``from_``
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

    limits : tuple[float, float]
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


# -- Data dashboard definitions ------------------------------------------------


class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

    def inverse(self, value):
        return 1.0 - self.__call__(value)


def _chunk_array(array, len_chunk):
    n_chunks = max(1, round(len(array) / len_chunk))
    return np.array_split(array, n_chunks)


def dashboard_particle_dataset(
    ds: xr.Dataset,
    title: str | None = None,
    phase_wavelength_chunk_size: int = 5,
    palette="muted",
):
    """
    Create a summary dashboard plot for a particle dataset. The generated plot
    displays the extinction coefficient, the albedo and phase function.

    Parameters
    ----------
    ds : Dataset
        Particle (aerosol, cloud, etc.) dataset.

    title : str, optional
        A title for the plot.

    phase_wavelength_chunk_size : int, default: 5
        The produced dashboard shows the phase function clustered in chunks on
        a number of polar plots set by this chunk size.

    palette : str, list, dict, or matplotlib.colors.Colormap, optional
        A Seaborn palette specification used to colour the phase function line
        plots.
    """
    df = ds.phase.isel(i=0, j=0, drop=True).to_dataframe()[["phase"]]
    df = df.sort_values(["w", "mu"]).reset_index()
    df["theta"] = np.arccos(df["mu"])
    phase_min = df["phase"].min()
    phase_max = df["phase"].max()
    # print(f"{phase_min = }, {phase_max = }")

    w_chunks = [
        (chunk.min(), chunk.max())
        for chunk in _chunk_array(np.unique(df["w"]), phase_wavelength_chunk_size)
    ]
    naxs_phase = len(w_chunks)
    ncols = 2
    nrows_phase = naxs_phase // ncols + (1 if naxs_phase % ncols else 0)
    height_ratios = [0.5, 0.5] + [1 for _ in range(nrows_phase)]

    fig, axs = plt.subplot_mosaic(
        [
            ["sigma_t", "sigma_t"],
            ["albedo", "albedo"],
        ]
        + [[f"phase_{i * 2}", f"phase_{i * 2 + 1}"] for i in range(nrows_phase)],
        height_ratios=height_ratios,
        figsize=(4 * ncols, 4 * sum(height_ratios)),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )

    for i_var, var in enumerate(["sigma_t", "albedo"]):
        gs = axs[var].get_gridspec()
        start = i_var * ncols
        stop = start + 1
        axs[var].remove()
        axs[var] = plt.subplot(SubplotSpec(gs, start, stop), projection=None)
        ds[var].plot(ax=axs[var], ls=":", marker=".")

    for i_phase, (wmin, wmax) in enumerate(w_chunks):
        _df = df.where((df["w"] >= wmin) & (df["w"] <= wmax)).dropna()
        ax = axs[f"phase_{i_phase}"]
        sns.lineplot(
            ax=ax,
            data=_df,
            x="theta",
            y="phase",
            hue="w",
            hue_norm=PiecewiseNorm(levels=_df["w"].values),
            legend="full",
            palette=palette,
        )
        ax.set_rlim([phase_min * 0.5, phase_max * 5.0])
        ax.set_rscale("log")
        ax.set_thetagrids(np.arange(0, 181, 45))
        ax.set_thetamax(180)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Deal with legend entries
        lgd_title = "wavelength"  # default value, overridden by metadata if any
        for field in ["long_name", "standard_name"]:
            if field in ds.w.attrs:
                lgd_title = ds.w.attrs[field]
                break

        if "units" in ds.w.attrs:
            lgd_title += f" [{ds.w.units}]"

        lgd = ax.legend(
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.1),
            title=lgd_title,
        )
        new_labels = [f"{w:.3g}" for w in np.unique(_df["w"].values)]
        for i, _ in enumerate(lgd.get_texts()):
            lgd.get_texts()[i].set_text(new_labels[i])

    # Clear empty axes
    for i in range(i_phase + 1, nrows_phase * 2):
        ax = axs[f"phase_{i}"]
        ax.remove()

    if title:
        fig.suptitle(title)

    return fig, axs


def absorption_database_spectral_coverage_mono(
    db: MonoAbsorptionDatabase, wrange=None, title=None
):
    df_index = db._index

    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

    color_cycle = cycler(color=sns.color_palette())

    for i, (filename, color) in enumerate(zip(df_index["filename"], color_cycle())):
        color = color["color"]
        spectral_coverage = db.spectral_coverage.loc[filename]
        w = spectral_coverage.index.values
        sns.histplot(x=w, color=color, ax=ax)

    ax.set_xlabel("Wavelength [nm]")
    if title:
        ax.set_title(title)
    sns.despine(offset=10)

    return fig, ax


def absorption_database_spectral_coverage_ckd(
    db: CKDAbsorptionDatabase, wrange=None, title=None
):
    df_index = db._index
    wmins = df_index["wl_min [nm]"]
    wmaxs = df_index["wl_max [nm]"]

    if wrange is not None:
        select = (wmins >= wrange[0]) & (wmaxs <= wrange[1])
        df_index = df_index.loc[select]

    wmins = df_index["wl_min [nm]"]
    wmaxs = df_index["wl_max [nm]"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 1))

    color_cycle = cycler(color=sns.color_palette())

    for i, (filename, color) in enumerate(zip(df_index["filename"], color_cycle())):
        color = color["color"]
        spectral_coverage = db.spectral_coverage.loc[filename]

        for _, (wmin, wmax) in spectral_coverage.iterrows():
            ax.fill_between([wmin, wmax], 0, 1, color=color, alpha=0.5)

        if i == 0:
            w = wmins.min()
            ax.scatter(w, 0.5, color=color, marker=8)
            ax.annotate(
                f"{w:4.2f}",
                (w, 0.5),
                color=color,
                textcoords="offset points",
                xytext=(-6, 0),
                ha="right",
                va="center",
            )

    else:
        w = wmaxs.max()
        ax.scatter(w, 0.5, color=color, marker=9)
        ax.annotate(
            f"{w:4.2f}",
            (w, 0.5),
            color=color,
            textcoords="offset points",
            xytext=(6, 0),
            ha="left",
            va="center",
        )

    ax.set_xlabel("Wavelength [nm]")
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    sns.despine(ax=ax, top=True, left=True, right=True, bottom=False)

    return fig, ax
