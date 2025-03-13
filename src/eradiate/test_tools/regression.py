from __future__ import annotations

import os
import typing
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import xarray as xr
from robot.api import logger
from scipy.stats import norm, t

from .. import data
from ..attrs import define, documented
from ..exceptions import DataError
from ..typing import PathLike
from ..util.misc import summary_repr


def regression_test_plots(
    ref: np.typing.ArrayLike,
    result: np.typing.ArrayLike,
    vza: np.typing.ArrayLike,
    metric: tuple[str, float],
    ref_var: t.Optional[np.typing.ArrayLike] = None,
    result_var: t.Optional[np.typing.ArrayLike] = None,
) -> t.Tuple[plt.Figure, t.List[t.List[plt.Axes]]]:
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

    ref_var: array-like, optional
        Variable variance for the reference data

    result_var : array-like, optional
        Variable variance for the simulation result

    Returns
    -------
    figure: Figure
        Pyplot Figure containing the report charts

    axes: list
        2x2 array of Axes included in the report Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")

    if ref_var is None or result_var is None:
        axes[0][0].plot(vza, ref, label="reference")
        axes[0][0].plot(vza, result, label="result")
    else:
        axes[0][0].errorbar(vza, ref, yerr=np.sqrt(ref_var), label="reference")
        axes[0][0].errorbar(vza, result, yerr=np.sqrt(result_var), label="result")

    axes[0][0].set_title("Reference and test result")
    handles, labels = axes[0][0].get_legend_handles_labels()

    axes[1][0].plot(vza, result - ref)
    axes[1][0].set_xlabel("VZA [deg]")
    axes[1][0].set_ylabel("BRF in principal plane [-]")
    axes[1][0].set_title("Absolute difference")

    axes[1][1].plot(vza, (result - ref) / ref)
    axes[1][1].set_title("Relative difference")

    axes[0][1].set_axis_off()
    axes[0][1].legend(handles=handles, labels=labels, loc="upper center")

    if metric[1] is None:
        axes[0][1].text(
            0.5,
            0.5,
            f'Metric "{metric[0]}" is not available',
            horizontalalignment="center",
        )
    else:
        axes[0][1].text(
            0.5, 0.5, f"{metric[0]}: {metric[1]:.4}", horizontalalignment="center"
        )

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
    If value is ``None``, the converter returns ``None``.
    If value is a path to a local file, load it as a dataset.
    If value is a path to a remote file, load it from the Eradiate data store.
    If value is a dataset, return it as is.
    If value is a path to a remote file but the data store raised a DataError,
    returns ``None``.
    """
    if value is None:
        return value

    try:
        if isinstance(value, (str, os.PathLike, bytes)):
            # Try to open a file if it is directly referenced
            if os.path.isfile(value):
                logger.info(
                    f'Loading reference dataset "{str(value)}" from disk',
                    also_console=True,
                )
                return xr.load_dataset(value)

            logger.info(
                f'Attempting to serve reference dataset "{str(value)}" from the data store',
                also_console=True,
            )
            logger.info(
                f"Fetched path: {data.data_store.fetch(value)}", also_console=True
            )

            # Try to serve the file from the data store
            return data.load_dataset(value)

        elif isinstance(value, xr.Dataset):
            return value

        else:
            raise ValueError(
                "Reference must be provided as a Dataset or a file path. "
                f"Got {type(value).__name__}"
            )

    except DataError:
        return None


@define
class RegressionTest(ABC):
    """
    Common interface for tests based on the comparison of a result array against
    reference values.
    """

    # Name used for the reference metric. Must be set be subclasses.
    METRIC_NAME: typing.ClassVar[str | None] = None

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
        "NetCDF file or a path to a resource served by the data store.",
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
        doc="Threshold for test evaluation",
        type="float",
        init_type="float",
    )

    archive_dir: Path = documented(
        attrs.field(kw_only=True, converter=Path),
        doc="Path to output artefact storage directory. Relative paths are "
        "interpreted with respect to the current working directory.",
        type=":class:`pathlib.Path`",
        init_type="path-like",
    )

    def __attrs_pre_init(self):
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
        archive_dir = os.path.abspath(self.archive_dir)

        # Absolute path where the reference will be stored
        fname_reference = os.path.join(archive_dir, fname + "-ref" + ext)
        # Absolute path where test output will be stored
        fname_result = os.path.join(archive_dir, fname + "-result" + ext)

        # if no valid reference is found, store the results as new ref and fail
        # the test
        if not self.reference:
            logger.info(
                f"No reference data was found! Storing test results as reference to {fname_reference}",
                also_console=True,
            )
            self._archive(self.value, fname_reference)
            self._plot(reference_only=True, metric_value=None)
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
            self._plot(reference_only=False, metric_value=None)
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
        self._plot(reference_only=False, metric_value=metric_value)

        return passed

    @abstractmethod
    def _evaluate(self, diagnostic_chart: bool = False) -> tuple[bool, float]:
        """
        Evaluate the test results and perform a comparison to the reference
        based on the criterion defined in the specialized class.

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

    def _plot(self, metric_value: float | None, reference_only: bool) -> None:
        """
        Create a plot to visualize the results of the test.
        If the ``reference only`` parameter is set, create only a simple plot
        visualizing the new reference data. Otherwise, create the more complex
        comparsion plots for the regression test.

        Parameters
        ----------
        metric_value : float or None
            The numerical value of the test metric.

        reference_only : bool
            If ``True``, create only a simple visualization of the computed
            data.
        """
        vza = np.squeeze(self.value.vza.values)
        val = np.squeeze(self.value[self.variable].values)

        fname = self.name
        ext = ".png"

        archive_dir = os.path.abspath(self.archive_dir)
        fname_plot = os.path.join(archive_dir, fname + ext)
        os.makedirs(os.path.dirname(fname_plot), exist_ok=True)

        if reference_only:
            figure = plt.figure(figsize=(8, 6))
            plt.plot(vza, val)
            plt.xlabel("VZA [deg]")
            plt.ylabel(f"{self.variable.upper()} in principal plane [-]")
            plt.title("Simulation result, can be used as new reference")

        else:
            ref = np.squeeze(self.reference[self.variable].values)

            figure, _ = regression_test_plots(
                ref, val, vza, (self.METRIC_NAME, metric_value)
            )

        html_svg = figure_to_html(figure)
        logger.info(html_svg, html=True, also_console=False)
        logger.info(f"Saving PNG report chart to {fname_plot}", also_console=True)

        plt.savefig(fname_plot)
        plt.close()


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
class IndependantStudentTTest(RegressionTest):
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
        if self.variable + "_var" not in self.reference:
            raise ValueError(
                f"The target reference for this T-test does not record the appropriate variance values, could not find the data array {self.variable + '_var'}"
            )
        if self.variable + "_var" not in self.value:
            raise ValueError(
                f"The target value for this T-test does not record the appropriate variance values, could not find the data array {self.variable + '_var'}"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_ref_np = self.reference[self.variable + "_var"].values.ravel()
        var_res_np = self.value[self.variable + "_var"].values.ravel()

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
        p_value = t.sf(np.abs(t_prim), dof) * 2

        passed = p_value > self.threshold

        if diagnostic_chart:
            plt.grid()
            start, end = t.ppf(0.0001, dof), t.ppf(0.9999, dof)
            if t_prim > start and t_prim < end:
                fx = np.linspace(-np.abs(t_prim), np.abs(t_prim), 100)
                fy = t(dof).pdf(fx)
                plt.fill_between(np.zeros((100,)), fy)
                plt.axvline(t.ppf(-self.threshold / 2.0, dof, color="red"))
                plt.axvline(t.ppf(self.threshold / 2.0, dof, color="red"))
            else:
                plt.axvline(t_prim, label="T value")
            x = np.linspace(start, end, 100)
            y = t(dof).pdf(x)
            plt.axvline(0.0, color="red", linestyle="--")
            plt.title("T-statistic")
            plt.legend(loc="upper left")
            ax2 = plt.twinx()
            ax2.plot(x, y, label="target T distribution form", color="black")
            ax2.legend(loc="upper right")
            ax2.set_ylim([0.0, max(y) * 1.1])
            chart = render_svg_chart()
            plt.close()
            logger.info(chart, html=True)

        logger.info(f"bias    = {bias_mean}", also_console=True)
        logger.info(f"s_p     = {s_p}", also_console=True)
        logger.info(f"t'      = {t_prim}", also_console=True)
        logger.info(f"dof     = {dof}", also_console=True)
        logger.info(f"p-value = {p_value}", also_console=True)
        logger.info(f"alpha   = {self.threshold}", also_console=True)

        return passed, p_value


@define
class PairedStudentTTest(RegressionTest):
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

    cov: np.typing.ArrayLike | float = documented(
        attrs.field(kw_only=True, default=0.0),
        doc="Covariance between observation, defaults to zero",
        type=np.typing.ArrayLike | float,
        init_type="float",
    )

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        if self.variable + "_var" not in self.reference:
            raise ValueError(
                f"The target reference for this T-test does not record the appropriate variance values, could not find the data array {self.variable + '_var'}"
            )
        if self.variable + "_var" not in self.value:
            raise ValueError(
                f"The target value for this T-test does not record the appropriate variance values, could not find the data array {self.variable + '_var'}"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_ref_np = self.reference[self.variable + "_var"].values.ravel()
        var_res_np = self.value[self.variable + "_var"].values.ravel()

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
        p_value = t.sf(np.abs(t_prim), dof) * 2

        passed = p_value > self.threshold

        if diagnostic_chart:
            plt.grid()
            start, end = t.ppf(0.0001, dof), t.ppf(0.9999, dof)
            if t_prim > start and t_prim < end:
                fx = np.linspace(-np.abs(t_prim), np.abs(t_prim), 100)
                fy = t(dof).pdf(fx)
                plt.fill_between(np.zeros((100,)), fy)
                plt.axvline(t.ppf(-self.threshold / 2.0, dof, color="red"))
                plt.axvline(t.ppf(self.threshold / 2.0, dof, color="red"))
            else:
                plt.axvline(t_prim, label="T value")
            x = np.linspace(start, end, 100)
            y = t(dof).pdf(x)
            plt.axvline(0.0, color="red", linestyle="--")
            plt.title("T-statistic")
            plt.legend(loc="upper left")
            ax2 = plt.twinx()
            ax2.plot(x, y, label="target T distribution form", color="black")
            ax2.legend(loc="upper right")
            ax2.set_ylim([0.0, max(y) * 1.1])
            chart = render_svg_chart()
            plt.close()
            logger.info(chart, html=True)

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

    def _evaluate(self, diagnostic_chart=False) -> tuple[bool, float]:
        if self.variable + "_var" not in self.value:
            raise ValueError(
                f"The target value for this Z-test does not record the appropriate variance values, could not find the data array {self.variable + '_var'}"
            )

        ref_np = self.reference[self.variable].values.ravel()
        result_np = self.value[self.variable].values.ravel()

        var_res_np = self.value[self.variable + "_var"].values.ravel()

        assert ref_np.shape == result_np.shape
        assert ref_np.shape == var_res_np.shape

        # Calculate Z-statistic
        z = (result_np - ref_np) / np.sqrt(var_res_np)

        # Calculate p-value of the two-tailed z-test null hypothesis
        p_values = norm.sf(np.abs(z)) * 2

        alpha_0 = 1.0 - (1.0 - self.threshold) ** (1.0 / result_np.size)
        accept_null = p_values > alpha_0

        passed = np.count_nonzero(accept_null) >= 0.9975 * result_np.size

        if diagnostic_chart:
            plt.grid()
            plt.hist(z, bins=50, label="Z values")
            x = np.linspace(-4.0, 4.0, 100)
            y = norm.pdf(x, 0.0, 1.0)
            plt.axvline(0.0, color="red", linestyle="--")
            plt.title("Z-statistic")
            plt.legend(loc="upper left")
            ax2 = plt.twinx()
            ax2.plot(x, y, label="target Z distribution form", color="black")
            ax2.legend(loc="upper right")
            ax2.set_ylim([0.0, max(y) * 1.1])
            chart = render_svg_chart()
            plt.close()
            logger.info(chart, html=True)

            logger.info(f"alpha_0 = {alpha_0}", also_console=True)

        logger.info(f"min p-value = {min(p_values)}", also_console=True)
        logger.info(f"max p-value = {max(p_values)}", also_console=True)
        logger.info(
            f"n passed    = {np.count_nonzero(accept_null)}/{0.99 * result_np.size}",
            also_console=True,
        )
        logger.info(f"alpha_1     = {self.threshold}", also_console=True)
        logger.info(f"alpha_0     = {alpha_0}", also_console=True)

        return passed, min(p_values)
