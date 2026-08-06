"""
Statistical criteria used to compare a simulation result against a reference,
and the charts that make a verdict readable.

This module is deliberately free of any I/O and of any dependency on Pytest:
reference data management (locating references, regenerating them, archiving
artefacts) is handled by the ``dataset_regression`` fixture, see
:mod:`eradiate.test_tools.fixtures`.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import attrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstats
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from ..attrs import define, documented

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class RegressionTestFailure(AssertionError):
    """
    Raised when a regression test does not pass, *i.e.* when the comparison
    against the reference fails.

    Deriving from :class:`AssertionError` is what makes the ``--force-regen``
    workflow work: :mod:`pytest_regressions` keys its reference regeneration on
    a failing comparison raising an assertion. It also keeps the failure
    distinct from the :class:`ValueError`\\ s the framework raises for malformed
    data, so that a caller can tell a failed comparison from a broken one.
    """


def sidak_family_p_value(p_min: float, n: int) -> float:
    """
    Aggregate ``n`` paired p-values into a single family-wise p-value using the
    Šidák correction, *i.e.* the probability of observing a minimum p-value at
    least as extreme as ``p_min`` among ``n`` independent comparisons.

    Parameters
    ----------
    p_min : float
        Smallest of the paired p-values.

    n : int
        Number of paired comparisons.

    Returns
    -------
    float
        Family-wise p-value, directly comparable with the test threshold: it
        exceeds the threshold if and only if every paired p-value exceeds the
        Šidák-corrected per-comparison level
        ``1 - (1 - threshold) ** (1 / n)``.

    Notes
    -----
    Computed as ``-expm1(n * log1p(-p_min))`` rather than
    ``1 - (1 - p_min) ** n`` to remain accurate for small ``p_min``.
    """
    if p_min >= 1.0:  # log1p(-1) is -inf: the limit is 1, computed without warning
        return 1.0

    return float(-np.expm1(n * np.log1p(-p_min)))


def _check_shape(array: ArrayLike, other: ArrayLike, variable: str, label: str) -> None:
    """
    Raise a :class:`ValueError` if `other` does not have the same shape as
    `array`, `label` naming what it is (*e.g.* ``"reference"``).
    """
    if np.shape(array) != np.shape(other):
        raise ValueError(
            f"Result and {label} data for variable '{variable}' do not have "
            f"the same shape! Got: {np.shape(array)}, {np.shape(other)}"
        )


def vza_dim(ds: xr.Dataset) -> str | None:
    """
    Name of the dimension the plots use as their x axis.

    ``vza`` is usually a non-dimension coordinate: a measure yields
    ``vza(x_index, y_index)`` and the angular sweep lives on ``x_index``. Only
    the datasets built by hand in the unit tests have a ``vza`` dimension.

    Parameters
    ----------
    ds : Dataset
        Dataset to inspect.

    Returns
    -------
    str or None
        The single non-degenerate dimension of the ``vza`` coordinate, or its
        first dimension if all of them are of length 1. ``None`` if the dataset
        carries no ``vza`` coordinate at all, *e.g.* when it only holds
        hemispherical quantities.
    """
    if "vza" not in ds:
        return None

    vza = ds["vza"]
    sweep = [d for d in vza.dims if vza.sizes[d] > 1]
    return str(sweep[0] if sweep else vza.dims[0])


def plot_x_axis(ds: xr.Dataset, variable: str) -> tuple[str, np.ndarray, str]:
    """
    Dimension of `variable` held out as the x axis of the comparison charts,
    together with the values and the label to plot it with.

    Parameters
    ----------
    ds : Dataset
        Dataset holding `variable`.

    variable : str
        Name of the plotted variable.

    Returns
    -------
    x_dim : str
        Dimension to hold out, to be passed to :func:`hue_from_extra_dims`.

    values : ndarray
        Values of the x axis.

    label : str
        Label of the x axis.

    Notes
    -----
    Variables sampled along the viewing direction are plotted against VZA.
    Hemispherical quantities (*e.g.* ``bhr``) have no VZA dimension — and a
    dataset holding only such quantities has no ``vza`` coordinate either: they
    are plotted against their own longest dimension, which usually leaves a
    single point.
    """
    da = ds[variable]

    x_dim = vza_dim(ds)
    if x_dim is not None and x_dim in da.dims:
        return x_dim, np.atleast_1d(np.squeeze(ds["vza"].values)), "VZA [deg]"

    x_dim = str(max(da.dims, key=lambda d: da.sizes[d]))
    values = da[x_dim].values if x_dim in da.coords else np.arange(da.sizes[x_dim])
    return x_dim, np.atleast_1d(values), x_dim


def hue_from_extra_dims(
    da: xr.DataArray, x_dim: str
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """
    Flatten every dimension of ``da`` except `x_dim` into a single one, meant to
    be mapped to colour by :func:`regression_test_plots`.

    Parameters
    ----------
    da : DataArray
        Data to flatten.

    x_dim : str
        Dimension held out, plotted along the x axis. See :func:`vza_dim`.

    Returns
    -------
    values : ndarray
        ``(n_x,)`` if `x_dim` is the only dimension left after squeezing,
        ``(n_hue, n_x)`` otherwise.

    hue : ndarray or None
        Coordinate values of the flattened dimension, ``None`` if there is none.

    hue_label : str or None
        Name of the flattened dimension, ``None`` if there is none.
    """
    # Squeeze length-1 dimensions, except `x_dim`
    # which always has to be preserved so it can be plotted
    da = da.squeeze([d for d in da.dims if d != x_dim and da.sizes[d] == 1], drop=True)

    extra = [str(d) for d in da.dims if d != x_dim]
    if not extra:
        return da.values, None, None

    da = da.transpose(*extra, x_dim)
    values = da.values.reshape(-1, da.sizes[x_dim])

    if len(extra) == 1 and extra[0] in da.coords:
        return values, np.asarray(da[extra[0]].values, dtype=float), extra[0]

    # No coordinate, or several dimensions stacked: the hue is an ordinal index
    # (fallback)
    return values, np.arange(values.shape[0], dtype=float), " x ".join(extra)


def annotate_panel(ax: Axes, text: str) -> None:
    """
    Label a panel with an annotation placed at the top centre of its data area
    (used to replace a title, saving vertical space).

    Parameters
    ----------
    ax : Axes
        Axes to annotate.

    text : str
        Annotation text.
    """

    ax.annotate(
        text,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, -6.0),
        textcoords="offset points",
        horizontalalignment="center",
        verticalalignment="top",
    )


def regression_test_plots(
    ref: ArrayLike,
    result: ArrayLike,
    vza: ArrayLike,
    metric: tuple[str, float],
    ref_var: ArrayLike | None = None,
    result_var: ArrayLike | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hue: ArrayLike | None = None,
    hue_label: str | None = None,
    diagnostic: Callable[[Axes], None] | None = None,
) -> tuple[Figure, list[list[Axes]]]:
    """
    Create regression test report plots. Plot errorbars if both ref_var and
    result_var are set.

    Parameters
    ----------
    ref : array-like
        Variable values for the reference data. Shape ``(n_vza,)``, or
        ``(n_hue, n_vza)`` if `hue` is set.

    result : array-like
        Variable values for the simulation result. Same shape as `ref`.

    vza : array-like
        VZA values for plotting

    metric : tuple
        A tuple of the form (metric name, value) to be added to the plots.

    ref_var : array-like, optional
        Variable variance for the reference data. Ignored if `hue` is set.

    result_var : array-like, optional
        Variable variance for the simulation result. Ignored if `hue` is set.

    xlabel, ylabel : str or None
        Labels applied to the x and y axes of the plot.

    hue : array-like, optional
        Coordinate values of an extra dimension, mapped to colour: one line per
        value on each data panel, reference dashed and result solid.

    hue_label : str or None
        Label of the `hue` colour bar.

    diagnostic : callable, optional
        Callback drawing a test-specific diagnostic chart on the fourth panel,
        called as ``diagnostic(ax)``. If unset, that panel is left blank.

    Returns
    -------
    figure: Figure
        Matplotlib Figure containing the report charts

    axes: list
        2×2 array of Axes included in the report Figure
    """
    try:
        import colorcet as cc
    except ImportError as e:
        raise ImportError(
            "Hue-encoded regression plots require the colorcet package"
        ) from e

    ref = np.atleast_2d(ref)
    result = np.atleast_2d(result)

    if hue is None:
        # Single slice, default colour cycle, solid lines: reference and result
        # are told apart by colour
        styles = [{}]
        ref_style = {}
    else:
        # Extra dimension is an ordered physical coordinate, hence a
        # sequential colormap. Reference and result share a colour and are told
        # apart by linestyle, so they stay comparable slice by slice.
        hue = np.asarray(hue, dtype=float)
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        cmap = cc.cm["isoluminant_cgo_70_c39"]
        styles = [{"color": cmap(norm(value))} for value in hue]
        ref_style = {"linestyle": "--"}
        # N overlaid errorbar families are unreadable
        ref_var = result_var = None

    # A hemispherical quantity reduces to a single point, which no line renders
    if np.size(vza) == 1:
        styles = [{"marker": "o", **style} for style in styles]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")

    ax = axes[0, 0]
    ax.set_ylabel(ylabel)
    for i, style in enumerate(styles):
        label = "reference" if i == 0 else None
        if ref_var is None:
            ax.plot(vza, ref[i], label=label, **style, **ref_style)
        else:
            ax.errorbar(vza, ref[i], yerr=np.sqrt(ref_var), label=label)

    for i, style in enumerate(styles):
        label = "result" if i == 0 else None
        if result_var is None:
            ax.plot(vza, result[i], label=label, **style)
        else:
            ax.errorbar(vza, result[i], yerr=np.sqrt(result_var), label=label)

    handles, labels = ax.get_legend_handles_labels()
    if hue is not None:
        # Rewrite legend to use black line color when hue coordinate is present
        handles = [
            Line2D([], [], color="black", linestyle=handle.get_linestyle())
            for handle in handles
        ]
    ax.legend(handles=handles, labels=labels)

    ax = axes[1, 0]
    for i, style in enumerate(styles):
        ax.plot(vza, result[i] - ref[i], **style)
    annotate_panel(ax, "absolute difference")

    ax = axes[1, 1]
    for i, style in enumerate(styles):
        ax.plot(vza, (result[i] - ref[i]) / ref[i], **style)
    annotate_panel(ax, "relative difference")

    # The fourth panel hosts the diagnostic chart, if the test provides one
    ax = axes[0, 1]
    if diagnostic is None:
        ax.set_axis_off()
    else:
        diagnostic(ax)
    ax.set_title(
        f'Metric "{metric[0]}" is not available'
        if metric[1] is None
        else f"{metric[0]}: {metric[1]:.4}",
    )

    # Colorbar on top of the data panel, in place of its title
    if hue is not None:
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[0][0],
            label=hue_label,
            location="top",
            pad=-0.05,
        )
        # Colorbar is continuous, data is not: mark sampled coordinates
        cbar.ax.vlines(
            hue, *cbar.ax.get_ylim(), colors="white", linestyles="--", linewidths=0.8
        )

    for i, j, _xlabel in [(0, 0, xlabel), (1, 0, xlabel), (1, 1, xlabel)]:
        ax = axes[i][j]
        if _xlabel is not None:
            ax.set_xlabel(_xlabel)

    return fig, axes


@define
class RegressionTestOutcome:
    """
    Verdict of a single :meth:`.RegressionTest.evaluate` call, together with
    everything a caller needs to report it.
    """

    passed: bool = documented(
        attrs.field(converter=bool),
        doc="``True`` iff the test passed.",
        type="bool",
    )

    metric_name: str = documented(
        attrs.field(),
        doc="Name of the test metric.",
        type="str",
    )

    metric_value: float | None = documented(
        attrs.field(),
        doc="Value of the test metric, ``None`` if it could not be computed.",
        type="float or None",
    )

    threshold: float = documented(
        attrs.field(),
        doc="Threshold the metric was compared with.",
        type="float",
    )

    variable: str = documented(
        attrs.field(),
        doc="Name of the tested data variable.",
        type="str",
    )

    details: dict[str, Any] = documented(
        attrs.field(factory=dict),
        doc="Intermediate quantities that document the decision, rendered as "
        "report lines by the caller.",
        type="dict",
        default="{}",
    )

    warnings: list[str] = documented(
        attrs.field(factory=list),
        doc="Messages the caller should surface as warnings, *e.g.* about "
        "degraded reference data.",
        type="list of str",
        default="[]",
    )

    diagnostic_data: dict[str, Any] = documented(
        attrs.field(factory=dict, repr=False),
        doc="Data consumed by :meth:`.RegressionTest.plot_diagnostic` to draw "
        "the diagnostic panel of the comparison chart. Empty when the test has "
        "no diagnostic.",
        type="dict",
        default="{}",
    )

    def __str__(self) -> str:
        lines = [
            "Test passed" if self.passed else "Test did not pass",
            f"Metric value: {self.metric_name} = {self.metric_value}",
            f"Metric threshold: {self.threshold}",
            f"Variable: {self.variable}",
        ]
        lines.extend(f"{key} = {value}" for key, value in self.details.items())
        return "\n".join(lines)


@define
class RegressionTest(ABC):
    """
    Common interface for tests based on the comparison of a result dataset
    against a reference dataset.

    Instances are pure comparators: they hold the criterion, not the data.
    Call :meth:`evaluate` to get a verdict and :meth:`plot` to chart it.
    """

    # Name used for the reference metric. Must be set by subclasses.
    METRIC_NAME: ClassVar[str | None] = None

    # ``True`` if the test passes when the metric is *below* the threshold
    # (RMSE), ``False`` if it passes when the metric is *above* it (p-value).
    # Making the direction explicit is what lets `dim` aggregation pick the
    # worst slice without knowing the metric.
    METRIC_LOWER_IS_BETTER: ClassVar[bool | None] = None

    threshold: float = documented(
        attrs.field(converter=float),
        doc="Test metric threshold.",
        type="float",
        init_type="float",
    )

    variable: str = documented(
        attrs.field(kw_only=True, default="brf_srf"),
        doc="Tested variable.",
        type="str",
        init_type="str",
        default='"brf_srf"',
    )

    dim: str | None = documented(
        attrs.field(kw_only=True, default=None),
        doc="If set, evaluate the metric independently for each value of this "
        "dimension. The test then passes iff every slice passes, and the "
        "reported metric is that of the worst slice. This is stricter than "
        "comparing the flattened arrays in one go, and it is how datasets "
        "aggregating several independent measurements (*e.g.* one per "
        "wavelength) must be tested. Tests whose threshold is a false-positive "
        "rate rather than an absolute tolerance (*e.g.* :class:`.ZTest`) "
        "correct for the number of slices, so that `threshold` keeps its "
        "meaning; see :meth:`evaluate`.",
        type="str or None",
        init_type="str, optional",
        default="None",
    )

    def __attrs_pre_init__(self):
        if self.METRIC_NAME is None or self.METRIC_LOWER_IS_BETTER is None:
            raise TypeError(f"Unsupported test type {type(self).__name__}")

    def evaluate(
        self, result: xr.Dataset, reference: xr.Dataset
    ) -> RegressionTestOutcome:
        """
        Compare `result` against `reference` using this test's criterion.

        Parameters
        ----------
        result : Dataset
            Simulation result.

        reference : Dataset
            Reference data.

        Returns
        -------
        RegressionTestOutcome

        Raises
        ------
        ValueError
            If the data is malformed, *e.g.* if the shapes do not match or if a
            required data variable is missing. This is a broken comparison, not
            a failed one: a failed comparison is reported through the returned
            outcome.

        Notes
        -----
        When `dim` is set, each slice is evaluated on its own and the per-slice
        metrics are combined into the single value carried by the outcome. Two
        hooks control that combination: :meth:`_slice_threshold` sets the
        threshold each slice is evaluated against, and :meth:`_aggregate_metrics`
        turns the per-slice metrics back into one comparable with `threshold`.
        Their defaults simply take the worst slice, which is what an absolute
        tolerance such as an RMSE calls for. A test whose threshold is a
        false-positive rate must override both, otherwise judging N slices at
        level :math:`\\alpha` would let the overall false-positive rate grow to
        :math:`1 - (1 - \\alpha)^N`; see :class:`.ZTest`.
        """
        if self.dim is None:
            return self._evaluate(result, reference)

        # `sel()` needs `dim` to be an indexed dimension of both datasets. A
        # missing or scalar `dim` is malformed data (raising here is more
        # comprehensive than letting xarray fail later).
        for label, ds in [("result", result), ("reference", reference)]:
            if self.dim not in ds.dims or self.dim not in ds.coords:
                raise ValueError(
                    f"The {label} data has no indexed dimension '{self.dim}' "
                    "to evaluate the test criterion along"
                )

        coords = result[self.dim].values
        if not len(coords):
            raise ValueError(f"Dimension '{self.dim}' of the result data is empty")

        # Slices are evaluated against a possibly corrected threshold, so that
        # the details they report (accept/reject counts, per-comparison levels)
        # describe the criterion the verdict is actually based on.
        sliced = attrs.evolve(self, threshold=self._slice_threshold(len(coords)))
        outcomes = [
            sliced._evaluate(
                result.sel({self.dim: coord}), reference.sel({self.dim: coord})
            )
            for coord in coords
        ]

        # Report the worst slice: it is the one that decided the verdict, and
        # the one worth looking at when the test fails.
        metrics = [outcome.metric_value for outcome in outcomes]
        worst = int(
            np.argmax(metrics) if self.METRIC_LOWER_IS_BETTER else np.argmin(metrics)
        )

        metric = self._aggregate_metrics(metrics)

        return attrs.evolve(
            outcomes[worst],
            passed=(
                metric <= self.threshold
                if self.METRIC_LOWER_IS_BETTER
                else metric > self.threshold
            ),
            metric_value=metric,
            threshold=self.threshold,
            details={f"worst {self.dim}": coords[worst]} | outcomes[worst].details,
            # Deduplicate identical error messages while preserving order
            warnings=list(
                dict.fromkeys(msg for outcome in outcomes for msg in outcome.warnings)
            ),
        )

    def _slice_threshold(self, n_slices: int) -> float:
        """
        Threshold each slice is evaluated against when `dim` is set. The default
        is the test threshold itself, appropriate for an absolute tolerance.
        """
        return self.threshold

    def _aggregate_metrics(self, metrics: list[float]) -> float:
        """
        Combine per-slice metrics into a single value comparable with
        `threshold`. The default is the worst slice, appropriate for an absolute
        tolerance.
        """
        return max(metrics) if self.METRIC_LOWER_IS_BETTER else min(metrics)

    @abstractmethod
    def _evaluate(
        self, result: xr.Dataset, reference: xr.Dataset
    ) -> RegressionTestOutcome:
        """
        Apply the test criterion to a single pair of datasets, *i.e.* without
        the per-slice dispatch performed by :meth:`evaluate`.
        """
        pass

    def plot(
        self,
        result: xr.Dataset,
        reference: xr.Dataset,
        outcome: RegressionTestOutcome | None = None,
    ) -> tuple[Figure, Any]:
        """
        Draw the comparison chart: reference and result overlaid, their absolute
        and relative differences, and the test's diagnostic panel.

        Parameters
        ----------
        result : Dataset
            Simulation result.

        reference : Dataset
            Reference data.

        outcome : RegressionTestOutcome, optional
            Verdict to annotate the chart with. If unset, the metric is
            reported as unavailable and no diagnostic panel is drawn — which is
            what to do when the comparison itself raised.

        Returns
        -------
        figure : Figure

        axes : ndarray
            2×2 array of Axes.
        """
        x_dim, x, xlabel = plot_x_axis(result, self.variable)
        val, hue, hue_label = hue_from_extra_dims(result[self.variable], x_dim)
        ref, _, _ = hue_from_extra_dims(reference[self.variable], x_dim)

        return regression_test_plots(
            ref,
            val,
            x,
            (self.METRIC_NAME, None if outcome is None else outcome.metric_value),
            xlabel=xlabel,
            ylabel=self.variable,
            hue=hue,
            hue_label=hue_label,
            diagnostic=self._diagnostic_plotter(outcome),
        )

    def plot_noref(self, result: xr.Dataset) -> tuple[Figure, Any]:
        """
        Draw a chart of a result that has no reference to be compared with,
        *i.e.* a reference candidate.

        Parameters
        ----------
        result : Dataset
            Simulation result.

        Returns
        -------
        figure : Figure

        axes : Axes
        """
        x_dim, x, xlabel = plot_x_axis(result, self.variable)
        val, _, _ = hue_from_extra_dims(result[self.variable], x_dim)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # One line per slice of the extra dimensions, if any. The colour cycle
        # is enough here: this plot has no reference to compare against.
        ax.plot(x, np.atleast_2d(val).T, marker="o" if x.size == 1 else None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.variable)
        ax.set_title("Simulation result, can be used as new reference")

        return fig, ax

    def _diagnostic_plotter(
        self, outcome: RegressionTestOutcome | None
    ) -> Callable[[Axes], None] | None:
        """
        Bind :meth:`plot_diagnostic` to the data collected during evaluation,
        ready to be drawn on a panel of the comparison chart. Returns ``None``
        when there is no diagnostic to draw.
        """
        if outcome is None or not outcome.diagnostic_data:
            return None
        return functools.partial(self.plot_diagnostic, **outcome.diagnostic_data)

    def plot_diagnostic(self, ax: Axes, **diagnostic_data) -> None:
        """
        Draw more technical information about the test metric and decision
        process on a panel of the comparison chart. The diagnostic plot can help
        the user debug a failing test, or assess the test power and
        significance.

        Parameters
        ----------
        ax : Axes
            Axes to draw on.

        **diagnostic_data
            Variadic keyword arguments for the subclass implementation, taken
            from :attr:`.RegressionTestOutcome.diagnostic_data`.
        """

        raise NotImplementedError(
            f"{type(self)} does not implement a diagnostic plot method"
        )


@define
class RMSETest(RegressionTest):
    """
    Root-mean-square error test.

    The test passes iff the root mean squared error (RMSE) of the result data
    against the reference is lower than or equal to the given threshold.
    """

    METRIC_NAME = "rmse"
    METRIC_LOWER_IS_BETTER = True

    def _evaluate(
        self, result: xr.Dataset, reference: xr.Dataset
    ) -> RegressionTestOutcome:
        value_np = result[self.variable].values
        ref_np = reference[self.variable].values
        _check_shape(value_np, ref_np, self.variable, "reference")

        result_flat = np.array(value_np).flatten()
        ref_flat = np.array(ref_np).flatten()

        rmse = float(np.linalg.norm(result_flat - ref_flat) / np.sqrt(len(ref_flat)))

        return RegressionTestOutcome(
            passed=rmse <= self.threshold,
            metric_name=self.METRIC_NAME,
            metric_value=rmse,
            threshold=self.threshold,
            variable=self.variable,
        )


@define
class ZTest(RegressionTest):
    """
    Z-test with Šidák correction factor.

    Implement a Z-test, testing the significance of paired differences between
    a set of observations and a set of references. It considers the variance of
    both the observations and the reference, which are both Monte Carlo
    estimates: the standard error of their difference is
    :math:`\\sqrt{\\sigma^2_\\mathrm{result} + \\sigma^2_\\mathrm{reference}}`.
    The observation variance (``<variable>_var``) is mandatory; if the reference
    does not carry one, the test falls back to the observation variance alone
    and reports a warning. That fallback underestimates the standard error by up
    to a factor of :math:`\\sqrt{2}`, making the test conservative.

    Paired tests are aggregated into one p-value using a Šidák correction: the
    test passes if the null hypothesis is accepted for *every* pair at the
    corrected per-comparison level :math:`1 - (1 - \\alpha)^{1/n}`. The reported
    metric is the equivalent family-wise p-value (see
    :func:`sidak_family_p_value`), so that the test passes iff the metric
    exceeds the threshold.

    This paired Z-test requires an equal degree of freedom of the two groups.

    Notes
    -----
    The `dim` field is diagnostic only: the correction extends over the slices,
    so the verdict and the reported family-wise p-value are those of the
    un-sliced test. What `dim` adds is the identity of the worst slice, and
    per-slice details describing it. Judging each of :math:`N` slices at level
    :math:`\\alpha` instead would let the overall false-positive rate grow to
    :math:`1 - (1 - \\alpha)^N` — 40 % for ten wavelengths at
    :math:`\\alpha = 0.05` — which is why slices are evaluated at
    :math:`1 - (1 - \\alpha)^{1/N}` and their p-values combined back with
    :func:`sidak_family_p_value`. Slices of a single dimension all hold the same
    number of comparisons, which is what makes the two routes exactly
    equivalent.
    """

    METRIC_NAME = "Z-test family p-value"
    METRIC_LOWER_IS_BETTER = False

    def _slice_threshold(self, n_slices: int) -> float:
        # Šidák again, one level up: the slices form a family of their own.
        return 1.0 - (1.0 - self.threshold) ** (1.0 / n_slices)

    def _aggregate_metrics(self, metrics: list[float]) -> float:
        # Undoes the per-slice correction: combining N slice-wise family
        # p-values, each computed over n comparisons, yields the family p-value
        # over the n * N comparisons of the un-sliced test.
        return sidak_family_p_value(min(metrics), len(metrics))

    def plot_diagnostic(self, ax: Axes, z=None) -> None:
        """
        Diagnostic plot for a Z-test.

        Parameters
        ----------
        ax : Axes
            Axes to draw on.

        z : array-like
            Z-statistic for each pair of measurements.
        """

        ax.grid()
        ax2 = ax.twinx()

        ax.hist(z, bins=50, label="Z values")
        ax.axvline(0.0, color="red", linestyle="--")
        ax.legend(loc="upper left", fontsize="small")

        x = np.linspace(-4.0, 4.0, 100)
        y = spstats.norm.pdf(x, 0.0, 1.0)
        ax2.plot(x, y, label="target", color="black")
        ax2.legend(loc="upper right", fontsize="small")
        ax2.set_ylim([0.0, max(y) * 1.1])

    def _evaluate(
        self, result: xr.Dataset, reference: xr.Dataset
    ) -> RegressionTestOutcome:
        variable_var = self.variable + "_var"
        warnings: list[str] = []

        if variable_var not in result:
            raise ValueError(
                "The result data for this Z-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        ref_np = reference[self.variable].values.ravel()
        result_np = result[self.variable].values.ravel()

        var_res_np = result[variable_var].values.ravel()

        # Malformed data must raise (not assert because
        # `perform_regression_check` keys `--force-regen` on AssertionError,
        # meaning that an assert here would have a broken comparison silently
        # overwrite the reference)
        _check_shape(result_np, ref_np, self.variable, "reference")
        _check_shape(result_np, var_res_np, self.variable, f"'{variable_var}'")

        # Both datasets are Monte Carlo estimates, so the variance of their
        # difference is the sum of their variances. If reference is missing
        # variance, fall back to the result variance alone (conservative).
        if variable_var in reference:
            var_ref_np = reference[variable_var].values.ravel()
            _check_shape(
                result_np, var_ref_np, self.variable, f"reference '{variable_var}'"
            )
        else:
            warnings.append(
                f"The reference data for this Z-test has no '{variable_var}' "
                "data variable; falling back to the result variance alone. The "
                "test is conservative in this configuration. Regenerate the "
                "reference to compare both variances."
            )
            var_ref_np = 0.0

        # Calculate Z-statistic
        z = (result_np - ref_np) / np.sqrt(var_res_np + var_ref_np)

        # Calculate p-value of the two-tailed z-test null hypothesis
        p_values = spstats.norm.sf(np.abs(z)) * 2

        alpha_0 = 1.0 - (1.0 - self.threshold) ** (1.0 / result_np.size)
        accept_null = p_values > alpha_0

        return RegressionTestOutcome(
            passed=bool(np.all(accept_null)),
            metric_name=self.METRIC_NAME,
            metric_value=sidak_family_p_value(min(p_values), result_np.size),
            threshold=self.threshold,
            variable=self.variable,
            details={
                "min p-value": min(p_values),
                "max p-value": max(p_values),
                "n accepted": f"{np.count_nonzero(accept_null)}/{result_np.size}",
                "alpha_1": self.threshold,
                "alpha_0": alpha_0,
            },
            warnings=warnings,
            diagnostic_data={"z": z},
        )

    def plot(
        self,
        result: xr.Dataset,
        reference: xr.Dataset,
        outcome: RegressionTestOutcome | None = None,
    ) -> tuple[Figure, Any]:
        """
        Draw a comparison plot with reference and test data displayed together,
        with the result's Monte Carlo variance shown as error bars.
        """
        x_dim, x, xlabel = plot_x_axis(result, self.variable)
        value, hue, hue_label = hue_from_extra_dims(result[self.variable], x_dim)
        result_var, _, _ = hue_from_extra_dims(result[f"{self.variable}_var"], x_dim)
        ref, _, _ = hue_from_extra_dims(reference[self.variable], x_dim)

        return regression_test_plots(
            ref,
            value,
            x,
            (self.METRIC_NAME, None if outcome is None else outcome.metric_value),
            result_var=result_var,
            xlabel=xlabel,
            ylabel=self.variable,
            hue=hue,
            hue_label=hue_label,
            diagnostic=self._diagnostic_plotter(outcome),
        )
