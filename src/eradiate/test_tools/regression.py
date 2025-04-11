from __future__ import annotations

import os
import typing as t
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import xarray as xr
from robot.api import logger

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
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

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
    METRIC_NAME: t.ClassVar[str | None] = None

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
            logger.info(
                "Test passed" if passed else "Test did not pass", also_console=True
            )
            logger.info(
                f"Metric value: {self.METRIC_NAME} = {metric_value}", also_console=True
            )
            logger.info(f"Metric threshold: {self.threshold}", also_console=True)
            logger.info(f"Variable: {self.variable}", also_console=True)
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
    def _evaluate(self) -> tuple[bool, float]:
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

        if self.variable == "brf_srf":  # Handle spectral results
            val = np.squeeze(self.value.brf_srf.values)
        else:  # Handle monochromatic results
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
            if self.variable == "brf_srf":  # Handle spectral results
                ref = np.squeeze(self.reference.brf_srf.values)
            else:  # Handle monochromatic results
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

    METRIC_NAME = "p-value"

    def _evaluate(self) -> tuple[bool, float]:
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
