from __future__ import annotations

import os
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import ClassVar

import attrs
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import scipy.stats as spstats
import xarray as xr
from numpy.typing import ArrayLike
from robot.api import logger

from .. import fresolver
from ..attrs import define, documented
from ..exceptions import DataError
from ..typing import PathLike
from ..util.misc import summary_repr


def regression_test_plots(
    ref: ArrayLike,
    result: ArrayLike,
    vza: ArrayLike,
    metric: tuple[str, float],
    ref_var: ArrayLike | None = None,
    result_var: ArrayLike | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """
    Create regression test report plots. Plot errorbars if both ref_var and
    result_var are set.

    Parameters
    ----------
    ref : array-like
        Variable values for the reference data

    result : array-like
        Variable values for the simulation result

    vza : array-like
        VZA values for plotting

    metric : tuple
        A tuple of the form (metric name, value) to be added to the plots.

    ref_var : array-like, optional
        Variable variance for the reference data.

    result_var : array-like, optional
        Variable variance for the simulation result.

    xlabel, ylabel : str or None
        Labels applied to the x and y axes of the plot.

    Returns
    -------
    figure: Figure
        Pyplot Figure containing the report charts

    axes: list
        2x2 array of Axes included in the report Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")

    ax = axes[0][0]
    if ref_var is None:
        ax.plot(vza, ref, label="reference")
    else:
        ax.errorbar(vza, ref, yerr=np.sqrt(ref_var), label="reference")

    if result_var is None:
        ax.plot(vza, result, label="result")
    else:
        ax.errorbar(vza, result, yerr=np.sqrt(result_var), label="result")

    ax.set_title("Reference and test result")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1][0]
    ax.plot(vza, result - ref)
    ax.set_title("Absolute difference")

    ax = axes[1][1]
    ax.plot(vza, (result - ref) / ref)
    ax.set_title("Relative difference")

    ax = axes[0][1]
    ax.set_axis_off()
    ax.legend(handles=handles, labels=labels, loc="upper center")

    if metric[1] is None:
        ax.text(
            0.5,
            0.5,
            f'Metric "{metric[0]}" is not available',
            horizontalalignment="center",
        )
    else:
        ax.text(0.5, 0.5, f"{metric[0]}: {metric[1]:.4}", horizontalalignment="center")

    for i, j in [[0, 0], [1, 0], [1, 1]]:
        ax = axes[i][j]
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    return fig, axes


def figure_to_html(fig: plt.Figure) -> str:
    """
    Render a figure in HTML format

    Returns a string containing the rendered HTML. The root tag is a <svg> one.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to render in HTML.

    Returns
    -------
    str
        Rendered HTML <svg> tag with styling.
    """

    str_i = StringIO()
    fig.savefig(str_i, format="svg", transparent=True, bbox_inches="tight")
    fig.canvas.draw_idle()
    svg = str_i.getvalue()

    # Include some CSS in the SVG to render nicely in Robot report's dark and
    # light modes
    return "\n".join(
        [
            "<svg",
            'version="1.1"',
            'baseProfile="full"',
            'width="810" height="540" viewBox="0 0 810 540"'
            'xmlns="http://www.w3.org/2000/svg">',
            "<style>",
            "    path {",
            "        fill: var(--text-color);",
            "        stroke: var(--text-color);",
            "    }",
            "</style>",
            svg,
            "</svg>",
        ]
    )


def reference_converter(
    value: PathLike | xr.Dataset | None,
) -> xr.Dataset | None:
    """
    A converter for handling the reference data attribute.

    Parameters
    ----------
    value : path-like or Dataset or None
        Path to the reference dataset file or dataset identifier or Dataset or
        None.

    Raises
    ------
    ValueError
        If the reference data is not a valid dataset.

    Returns
    -------
    xr.Dataset or None
        The reference dataset.

    Notes
    -----
    * ``None`` and datasets are passed through.
    * If ``value`` is a path, resolve it with the path resolver and try to load it.
    * If ``value`` is a path to a file that does not exist, return ``None``.
    """
    if value is None:
        return value

    if isinstance(value, xr.Dataset):
        return value

    if isinstance(value, (str, os.PathLike, bytes)):
        try:
            logger.info(f'Looking up "{str(value)}" on disk', also_console=True)
            fname = fresolver.resolve(value)
            logger.info(f"Resolved path: {fname}", also_console=True)
            return xr.load_dataset(fname)

        except (DataError, FileNotFoundError):
            pass

        # File not found: most likely means reference data does not exist
        return None

    raise ValueError(
        "Reference must be provided as a Dataset, a file path or None. "
        f"Got {type(value).__name__}"
    )


@define
class RegressionTest(ABC):
    """
    Common interface for tests based on the comparison of a result array against
    reference values.
    """

    # Name used for the reference metric. Must be set be subclasses.
    METRIC_NAME: ClassVar[str | None] = None

    name: str = documented(
        attrs.field(validator=attrs.validators.instance_of(str)),
        doc="Test case name.",
        type="str",
        init_type="str",
    )

    value: xr.Dataset = documented(
        attrs.field(
            validator=attrs.validators.instance_of(xr.Dataset),
            repr=summary_repr,
        ),
        doc="Simulation result. Must be specified as a dataset.",
        type=":class:`xarray.Dataset`",
        init_type=":class:`xarray.Dataset`",
    )

    reference: xr.Dataset | None = documented(
        attrs.field(
            default=None,
            converter=reference_converter,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(xr.Dataset)
            ),
        ),
        doc="Reference data. Can be specified as an xarray dataset, a path to a "
        "NetCDF file or a path to a resource.",
        type=":class:`xarray.Dataset` or None",
        init_type=":class:`xarray.Dataset` or path-like, optional",
        default="None",
    )

    variable: str = documented(
        attrs.field(kw_only=True, default="brf_srf"),
        doc="Tested variable",
        type="str",
        init_type="str",
        default="brf_srf",
    )

    threshold: float = documented(
        attrs.field(kw_only=True),
        doc="Test metric threshold",
        type="float",
        init_type="float",
    )

    archive_dir: Path = documented(
        attrs.field(kw_only=True, converter=lambda x: Path(x).resolve()),
        doc="Path to output artefact storage directory. Relative paths are "
        "interpreted with respect to the current working directory.",
        type=":class:`pathlib.Path`",
        init_type="path-like",
    )

    def __attrs_pre_init__(self):
        if self.METRIC_NAME is None:
            raise TypeError(f"Unsupported test type {type(self).__name__}")

    def run(self) -> bool:
        """
        This method controls the execution steps of the regression test:

        * handle missing reference data
        * catch errors during text evaluation
        * create the appropriate plots and data archives

        Returns
        -------
        bool
            Result of the test criterion comparison.
        """

        logger.info(f"Regression test {self.name} results:", also_console=True)

        fname = self.name
        ext = ".nc"
        archive_dir = self.archive_dir

        fname_reference = archive_dir / f"{fname}-ref{ext}"
        fname_result = archive_dir / f"{fname}-result{ext}"

        # if no valid reference is found, store the results as new ref and fail
        # the test
        if not self.reference:
            logger.info(
                "No reference data found. Storing test results to "
                f"{fname_reference}. This can be the new reference.",
                also_console=True,
            )
            self._archive(self.value, fname_reference)
            self._plot(metric_value=None, noref=True)
            return False

        # else (we have a reference value), evaluate the test metric
        try:
            passed, metric_value = self._evaluate()
            msg = "\n".join(
                [
                    "Test passed" if passed else "Test did not pass",
                    f"Metric value: {self.METRIC_NAME} = {metric_value}",
                    f"Metric threshold: {self.threshold}",
                    f"Variable: {self.variable}",
                ]
            )
            logger.info(msg, also_console=True)

        except Exception as e:
            logger.info(
                "An exception occurred during test evaluation!", also_console=True
            )
            self._plot(noref=False, metric_value=None)
            raise e

        # we got a metric: report the results in the archive directory
        logger.info(
            f"Saving current output dataset to {fname_result}", also_console=True
        )
        self._archive(self.value, fname_result)
        logger.info(
            f"Saving reference dataset locally to {fname_reference}", also_console=True
        )
        self._archive(self.reference, fname_reference)
        self._plot(noref=False, metric_value=metric_value)

        return passed

    @abstractmethod
    def _evaluate(self, diagnostic_chart: bool = False) -> tuple[bool, float]:
        """
        Evaluate the test results and perform a comparison to the reference
        based on the criterion defined in the specialized class.

        Parameters
        ----------
        diagnostic_chart : bool, optional
            If ``True``, append a diagnostic chart to the test report.

        Returns
        -------
        passed : bool
            ``True`` iff the test passed.

        metric_value : float
            The value of the test metric.
        """
        pass

    def _archive(self, dataset: xr.Dataset, fname_output: PathLike) -> None:
        """
        Create an archive file for test result and reference storage
        """
        os.makedirs(os.path.dirname(fname_output), exist_ok=True)
        dataset.to_netcdf(fname_output)

    def _plot(self, metric_value: float | None, noref: bool) -> None:
        """
        Create a plot to visualize the results of the test.
        If the ``reference only`` parameter is set, create only a simple plot
        visualizing the new reference data. Otherwise, create the more complex
        comparison plots for the regression test.

        Parameters
        ----------
        metric_value : float or None
            The numerical value of the test metric.

        noref : bool
            If ``True``, create only a simple visualization of the computed
            data.
        """

        fname = self.name
        ext = ".png"

        archive_dir = self.archive_dir
        fname_plot = archive_dir / f"{fname}{ext}"
        fname_plot.parent.mkdir(parents=True, exist_ok=True)

        if noref:
            figure, _ = self._plot_noref()
        else:
            figure, _ = self._plot_ref(metric_value)

        html_svg = figure_to_html(figure)
        logger.info(html_svg, html=True, also_console=False)
        logger.info(f"Saving PNG report chart to {fname_plot}", also_console=True)

        plt.savefig(fname_plot)
        plt.close()

    def _plot_noref(self):
        """
        Draw a simple plot when no reference data is available.
        """
        vza = np.squeeze(self.value.vza.values)
        val = np.squeeze(self.value[self.variable].values)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(vza, val)
        ax.set_xlabel("VZA [deg]")
        ax.set_ylabel(self.variable)
        ax.set_title("Simulation result, can be used as new reference")

        return fig, ax

    def _plot_ref(self, metric_value: float | None = None):
        """
        Draw a comparison plot with reference and test data displayed together.
        """
        vza = np.squeeze(self.value.vza.values)
        val = np.squeeze(self.value[self.variable].values)
        ref = np.squeeze(self.reference[self.variable].values)

        return regression_test_plots(
            ref,
            val,
            vza,
            (self.METRIC_NAME, metric_value),
            xlabel="VZA [deg]",
            ylabel=self.variable,
        )

    def _plot_diagnostic(self, **diagnostic_info) -> None:
        """
        Create an additional plot to display more technical information about
        the test metrics and decision process. The diagnostic plot can help the
        user debug a failing test, or to assess the test power and significance.
        This plot is output directly to the robotframework report.

        Parameters:
        -----------
        **diagnostic_info: dict
            Variadic keyword arguments for the subclasses implementation
        """

        raise NotImplementedError(
            f"{type(self)} does not implement a diagnostic plot method"
        )


@define
class RMSETest(RegressionTest):
    """
    This class implements a simple test based on the root mean squared
    error (RMSE) of a result array with respect to the reference data.

    The test will pass if the computed root mean squared deviation between
    the result and reference is smaller or equal to the given threshold.
    """

    METRIC_NAME = "rmse"

    def _evaluate(self) -> tuple[bool, float]:
        value_np = self.value.brf.values
        ref_np = self.reference.brf.values
        if np.shape(value_np) != np.shape(ref_np):
            raise ValueError(
                f"Result and reference do not have the same shape! "
                f"Got: {np.shape(value_np)}, {np.shape(ref_np)}"
            )

        result_flat = np.array(value_np).flatten()
        ref_flat = np.array(ref_np).flatten()

        rmse = np.linalg.norm(result_flat - ref_flat) / np.sqrt(len(ref_flat))
        return rmse <= self.threshold, rmse


@define
class Chi2Test(RegressionTest):
    """
    This class implements a statistical test for the regression testing
    campaign, based on Pearson's Chi-squared test.
    https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

    It determines the probability for the reference and the test result
    following the same distribution.

    This test will pass if the computed p-value is strictly larger than the
    given threshold.
    """

    # The algorithm is adapted from Mitsuba's testing framework.

    METRIC_NAME = "X² p-value"

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        ref_np = self.reference.brf.values

        result_np = self.value.brf.values
        histo_bins = np.linspace(ref_np.min(), ref_np.max(), 20)
        histo_ref = np.histogram(ref_np, histo_bins)[0]
        histo_res = np.histogram(result_np, histo_bins)[0]

        # sorting both histograms following the ascending frequencies in
        # the reference. Algorithm from:
        # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
        histo_ref_sorted, histo_res_sorted = zip(
            *sorted(zip(histo_ref, histo_res), key=lambda x: x[0])
        )

        from mitsuba.python.math import rlgamma

        chi2val, dof, pooled_in, pooled_out = mi.math.chi2(
            histo_res_sorted, histo_ref_sorted, 5
        )
        p_value = 1.0 - rlgamma(dof / 2.0, chi2val / 2.0)

        return p_value > self.threshold, p_value


@define
class AbstractStudentTTest(RegressionTest):
    """
    Abstract Student's T-Test
    =========================

    Implement diagnostic chart common to subclassing T-test implementations.
    """

    def _plot_diagnostic(self, dof=None, t_prim=None) -> None:
        """
        Diagnostic chart for an Independant Student's T-test

        Parameters:
        -----------
        dof: int
            Degrees of Freedom
        t_prim: float
            t' statistic issued from the test
        """
        fig, ax = plt.subplots()
        ax.grid()
        ax2 = ax.twinx()

        start, end = spstats.t.ppf(0.0001, dof), spstats.t.ppf(0.9999, dof)

        if (t_prim > start) and (t_prim < end):
            fx = np.linspace(-np.abs(t_prim), np.abs(t_prim), 100)
            fy = spstats.t(dof).pdf(fx)
            ax.fill_between(np.zeros((100,)), fy)
            ax.axvline(spstats.t.ppf(-self.threshold / 2.0, dof), color="red")
            ax.axvline(spstats.t.ppf(self.threshold / 2.0, dof), color="red")
        else:
            ax.axvline(t_prim, label="T value")

        ax.axvline(0.0, color="red", linestyle="--")
        ax.set_title("T-statistic")

        x = np.linspace(start, end, 100)
        y = spstats.t(dof).pdf(x)

        ax2.plot(x, y, label="target T distribution form", color="black")
        ax2.legend(loc="upper right")
        ax2.set_ylim([0.0, max(y) * 1.1])

        chart = figure_to_html(fig)
        plt.close(fig)

        logger.info(chart, html=True)


@define
class IndependantStudentTTest(AbstractStudentTTest):
    """
    Independant Student's T-test
    ============================

    This implementation of a Student's T-test is following the assumption of
    independance of the two groups that are tested. The bias of the mean values
    of the two groups is assumed to be the result of chance under the null
    hypothesis. It is a two-tailed test.

    It is less sensitive to outliers than the paired Student's T-test.
    """

    METRIC_NAME = "T-test p-value"

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        variable_var = self.variable + "_var"

        if variable_var not in self.reference:
            raise ValueError(
                "The reference data for this T-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        if variable_var not in self.value:
            raise ValueError(
                "The tested data for this T-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_ref_np = self.reference[variable_var].values.ravel()
        var_res_np = self.value[variable_var].values.ravel()

        # Calculate mean values over observations and associated variances
        R_res = np.mean(result_np)
        R_ref = np.mean(ref_np)
        var_R_res = np.sum(var_res_np) / var_res_np.size**2
        var_R_ref = np.sum(var_ref_np) / var_ref_np.size**2
        bias_mean = R_res - R_ref

        # Calculate T-statistic and associated degree of freedom of its
        # T-distribution using a pooled standard deviation

        s_p = np.sqrt(
            ((var_res_np.size - 1.0) * var_R_res + (var_ref_np.size - 1.0) * var_R_ref)
            / (var_res_np.size + var_ref_np.size - 2)
        )
        t_prim = bias_mean / (
            s_p * np.sqrt(1.0 / var_res_np.size + 1.0 / var_ref_np.size)
        )

        dof = (var_res_np.size + var_ref_np.size) - 2

        assert dof > 0

        # Calculate p-value of the two-tailed t-test using the T distribution
        # survival function for the null hypothesis.
        p_value = spstats.t.sf(np.abs(t_prim), dof) * 2

        passed = p_value > self.threshold

        if diagnostic_chart:
            self._plot_diagnostic(dof=dof, t_prim=t_prim)

        logger.info(f"bias    = {bias_mean}", also_console=True)
        logger.info(f"s_p     = {s_p}", also_console=True)
        logger.info(f"t'      = {t_prim}", also_console=True)
        logger.info(f"dof     = {dof}", also_console=True)
        logger.info(f"p-value = {p_value}", also_console=True)
        logger.info(f"alpha   = {self.threshold}", also_console=True)

        return passed, p_value


@define
class PairedStudentTTest(AbstractStudentTTest):
    """
    Paired Student's T-test
    =======================

    This implementation of a Student's T-test is following the assumption of
    paired samples within two groups that are tested. The mean of the bias
    between the paired values is assumed to be the result of chance under the
    null hypothesis. It is a two-tailed test.

    The paired test allow to introduce a covariance factor between the pairs.
    By default, this covariance is equal to zero, thus assuming independance of
    the two variables.

    Contrary to the independant Student's T-test, this paired version of the
    test requires an equal degree of freedom of the two groups.
    """

    METRIC_NAME = "paired T-test p-value"

    cov: np.ndarray | float = documented(
        attrs.field(kw_only=True, default=0.0),
        doc="Covariance between observation, defaults to zero",
        type="ndarray or float",
        init_type="array-like or float",
    )

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        variable_var = self.variable + "_var"

        if variable_var not in self.reference:
            raise ValueError(
                "The reference data for this T-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        if variable_var not in self.value:
            raise ValueError(
                "The tested data for this T-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_ref_np = self.reference[variable_var].values.ravel()
        var_res_np = self.value[variable_var].values.ravel()

        assert ref_np.shape == result_np.shape
        assert ref_np.shape == var_ref_np.shape
        assert ref_np.shape == var_res_np.shape

        # Calculate paired mean value and associated variance
        D_mean = np.mean(result_np - ref_np)
        var_D = (var_res_np + var_ref_np) - 2 * self.cov
        var_D_mean = np.sum(var_D) / var_D.size**2

        # Calculate T-statistic and associated degree of freedom of its
        # T-distribution
        t_prim = D_mean / (var_D_mean / np.sqrt(var_D.size))
        dof = var_D.size - 1

        assert dof > 0

        # Calculate p-value of the two-tailed t-test using the T distribution
        # survival function for the null hypothesis.
        p_value = spstats.t.sf(np.abs(t_prim), dof) * 2

        passed = p_value > self.threshold

        if diagnostic_chart:
            self._plot_diagnostic(dof=dof, t_prim=t_prim)

        logger.info(f"bias     = {D_mean}", also_console=True)
        logger.info(f"var mean = {var_D_mean}", also_console=True)
        logger.info(f"t'       = {t_prim}", also_console=True)
        logger.info(f"dof      = {dof}", also_console=True)
        logger.info(f"p-value  = {p_value}", also_console=True)
        logger.info(f"alpha    = {self.threshold}", also_console=True)

        return passed, p_value


@define
class ZTest(RegressionTest):
    """
    Z-Test with Šidák correction factor
    ===================================

    Implement a Z-test, testing the significance of paired differences between
    a set of observations and a set of references. It considers the observations
    variance.

    Paired tests are aggregated into one p-value using a Šidák correction. The
    test passes if the null hypothesis is accepted for at least 99.75% of the
    paired Z-tests

    This paired Z-test requires an equal degree of freedom of the two groups.
    """

    METRIC_NAME = "Z-test p-value"

    def _plot_diagnostic(self, z=None) -> None:
        """
        Diagnostic chart for a Z-test

        Parameters:
        -----------
        z: array-like
            Z-statistic for each pair of measurements
        """

        fig, ax = plt.subplots()
        ax.grid()
        ax2 = ax.twinx()

        ax.hist(z, bins=50, label="Z values")
        ax.axvline(0.0, color="red", linestyle="--")
        ax.set_title("Z-statistic")
        ax.legend(loc="upper left")

        x = np.linspace(-4.0, 4.0, 100)
        y = spstats.norm.pdf(x, 0.0, 1.0)
        ax2.plot(x, y, label="target Z distribution form", color="black")
        ax2.legend(loc="upper right")
        ax2.set_ylim([0.0, max(y) * 1.1])

        chart = figure_to_html(fig)
        plt.close(fig)

        logger.info(chart, html=True)

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        variable_var = self.variable + "_var"

        if variable_var not in self.value:
            raise ValueError(
                "The reference data for this Z-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_res_np = self.value[variable_var].values.ravel()

        assert ref_np.shape == result_np.shape
        assert ref_np.shape == var_res_np.shape

        # Calculate Z-statistic
        z = (result_np - ref_np) / np.sqrt(var_res_np)

        # Calculate p-value of the two-tailed z-test null hypothesis
        p_values = spstats.norm.sf(np.abs(z)) * 2

        alpha_0 = 1.0 - (1.0 - self.threshold) ** (1.0 / result_np.size)
        accept_null = p_values > alpha_0

        passed = np.count_nonzero(accept_null) >= 0.9975 * result_np.size

        if diagnostic_chart:
            self._plot_diagnostic(z=z)

        logger.info(f"min p-value = {min(p_values)}", also_console=True)
        logger.info(f"max p-value = {max(p_values)}", also_console=True)
        logger.info(
            f"n passed    = {np.count_nonzero(accept_null)}/{0.99 * result_np.size}",
            also_console=True,
        )
        logger.info(f"alpha_1     = {self.threshold}", also_console=True)
        logger.info(f"alpha_0     = {alpha_0}", also_console=True)

        return passed, min(p_values)

    def _plot_ref(self, metric_value: float | None = None):
        """
        Draw a comparison plot with reference and test data displayed together.
        """
        vza = np.squeeze(self.value.vza.values)
        result = np.squeeze(self.value[self.variable].values)
        result_var = np.squeeze(self.value[f"{self.variable}_var"].values)
        ref = np.squeeze(self.reference[self.variable].values)

        return regression_test_plots(
            ref,
            result,
            vza,
            (self.METRIC_NAME, metric_value),
            result_var=result_var,
            xlabel="VZA [deg]",
            ylabel=self.variable,
        )


@define
class SidakTTest(RegressionTest):
    """
    T-Test with Šidák correction factor
    ===================================

    Implement a T-test, testing the significance of paired differences between
    a set of observations and a set of references. It considers both the
    observations and reference variance.

    Paired tests are aggregated into one p-value using a Šidák correction. The
    test passes if the null hypothesis is accepted for at least 99.75% of the
    paired T-tests
    """

    METRIC_NAME = "Sidak T-test p-value"

    def _plot_diagnostic(self, t_prim=None) -> None:
        """
        Diagnostic chart for the T-test

        Parameters:
        -----------
        t_prim: array-like
            T-statistic for each pair of measurements
        """

        fig, ax = plt.subplots()
        ax.grid()
        ax2 = ax.twinx()

        start, end = spstats.norm.ppf(0.0001), spstats.norm.ppf(0.9999)

        ax.hist(t_prim, bins=50, label="T values")
        ax.axvline(0.0, color="red", linestyle="--")
        ax.set_title("T-statistic")
        ax.legend(loc="upper left")

        x = np.linspace(start, end, 100)
        y = spstats.norm.pdf(x)
        ax2.plot(x, y, label="target T distribution form", color="black")
        ax2.legend(loc="upper right")
        ax2.set_ylim([0.0, max(y) * 1.1])

        chart = figure_to_html(fig)
        plt.close(fig)

        logger.info(chart, html=True)

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        variable_var = self.variable + "_var"

        if variable_var not in self.reference:
            raise ValueError(
                "The reference data for this T-test does not contain expected "
                "appropriate variance values, could not find data variable "
                f"'{variable_var}'"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        assert ref_np.shape == result_np.shape

        var_ref_np = self.reference[variable_var].values.ravel()
        var_res_np = self.value[variable_var].values.ravel()

        assert var_ref_np.shape == var_res_np.shape

        # Calculate T-statistic
        t_prim = (result_np - ref_np) / np.sqrt(var_res_np + var_ref_np)

        # Calculate p-value of the two-tailed t-test using the T distribution
        # survival function for the null hypothesis that there is no difference
        # between the two mean distributions. It is assumed that the sample size
        # is large enough for the T distribution to converge to a normal one.
        p_values = spstats.norm.sf(np.abs(t_prim)) * 2

        # Calculate the Šidák correction
        alpha_0 = 1.0 - (1.0 - self.threshold) ** (1.0 / result_np.size)
        accept_null = p_values > alpha_0

        passed = np.count_nonzero(accept_null) >= 0.9975 * result_np.size

        if diagnostic_chart:
            self._plot_diagnostic(t_prim=t_prim)

        logger.info(f"min p-value = {min(p_values)}", also_console=True)
        logger.info(f"max p-value = {max(p_values)}", also_console=True)
        logger.info(
            f"n passed    = {np.count_nonzero(accept_null)}/{0.99 * result_np.size}",
            also_console=True,
        )
        logger.info(f"alpha_1     = {self.threshold}", also_console=True)
        logger.info(f"alpha_0     = {alpha_0}", also_console=True)

        return passed, min(p_values)
