"""
Extensions and helpers for tutorials.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from ..scenes.atmosphere import AbstractHeterogeneousAtmosphere
from ..spectral.index import SpectralIndex


def plot_sigma_t(
    *atmospheres: AbstractHeterogeneousAtmosphere,
    labels: list[str] | None = None,
    altitude_extent: tuple[float, float] | None = None,
    si: SpectralIndex | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Display the extinction coefficient of one or several atmosphere objects for
    a single spectral context.

    Parameters
    ----------
    *atmospheres : .AbstractHeterogeneousAtmosphere
        One or several atmosphere objects for which to plot the extinction
        coefficient.

    labels : list of strings, optional
        Labels associated with the passed atmosphere objects. If unset, no
        legend will be added to the plot.

    altitude_extent : tuple of float
        A (min, max) altitude pair (in km) to which the plot is restricted.

    si : .SpectralIndex, optional
        The spectral index at which the extinction coefficient is evaluated.
        If unset, a default spectral index is created using
        :meth:`.SpectralIndex.new`.

    show : bool, optional
        If ``True``, return ``None`` and display the plot. Otherwise, return a
        (``Figure``, ``Axes``) pair.
    """

    from matplotlib.ticker import ScalarFormatter

    from eradiate.units import to_quantity

    if si is None:  # Use the default spectral context (550 nm) if unspecified
        si = SpectralIndex.new()

    if labels is None:
        label_iter = iter([None for _ in atmospheres])
    else:
        label_iter = iter(labels)

    fig, ax = plt.subplots(1, 1)

    with plt.rc_context({"lines.linestyle": ":", "lines.marker": "."}):
        for atmosphere in atmospheres:
            altitude = to_quantity(atmosphere.eval_radprops(si=si).z_layer).m_as("km")
            sigma_t = to_quantity(atmosphere.eval_radprops(si=si).sigma_t).m_as("1/m")
            ax.plot(altitude, sigma_t, label=next(label_iter))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel("Extinction coefficient [1/m]")

    if labels is not None:
        fig.legend(
            bbox_to_anchor=(1.0, 0.5),
            loc="center left",
            borderaxespad=0.0,
        )

    if altitude_extent is not None:
        ax.set_xlim(altitude_extent)
    ax.set_xlabel("Altitude [km]")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax


def load_ipython_extension(ipython):
    """
    IPython extension for Eradiate tutorials. Should be loaded at the top of the
    tutorial notebook.

    It sets the Matplotlib style and prints the current date and Eradiate
    version as markdown.

    See Also
    --------
    :func:`eradiate.plot.set_style`

    Notes
    -----

    This extension should be loaded using the IPython magic:

    .. code::

       %load_ext eradiate.notebook.tutorials
    """

    import datetime

    from IPython.display import Markdown, display

    from .. import __version__
    from ..plot import set_style

    set_style()
    display(
        Markdown(
            f"*Last updated: {datetime.datetime.now():%Y-%m-%d %H:%M} "
            f"(eradiate v{__version__})*"
        )
    )
