from __future__ import annotations

import os
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import xarray as xr

from .. import data
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike
from ..util.misc import summary_repr


def regression_test_plots(
    ref: np.typing.ArrayLike,
    result: np.typing.ArrayLike,
    vza: np.typing.ArrayLike,
    filename: PathLike,
    metric: tuple[str, float],
) -> None:
    """
    Create regression test report plots.

    Parameters
    ----------
    ref : array-like
        BRF values for the reference data

    result : array-like
        BRF values for the simulation result

    vza : array-like
        VZA values for plotting

    filename : path-like
        Path to the output file for the plot

    metric : tuple
        A tuple of the form (metric name, value) to be added to the plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0][0].plot(vza, ref, label="reference")
    axes[0][0].plot(vza, result, label="result")
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
    axes[0][1].text(
        0.5, 0.5, f"{metric[0]}: {metric[1]:.4}", horizontalalignment="center"
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def reference_converter(
    value: os.PathLike | xr.Dataset | None,
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
                return xr.load_dataset(value)

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


@parse_docs
@attrs.define
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
            print("No reference data was found! Storing test results as reference.")
            self._archive(self.value, fname_reference)
            self._plot(reference_only=True, metric_value=None)
            return False

        # else (we have a reference value), evaluate the test metric
        try:
            passed, metric_value = self._evaluate()
        except Exception as e:
            print("An exception occurred during test execution!")
            self._plot(reference_only=False, metric_value=None)
            raise e

        # we got a metric: report the results in the archive directory
        self._archive(self.value, fname_result)
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
        print(f"Saving dataset to {fname_output}")
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

        if "brf_srf" in self.value.data_vars:  # Handle spectral results
            brf = np.squeeze(self.value.brf_srf.values)
        else:  # Handle monochromatic results
            brf = np.squeeze(self.value.brf.values)

        fname = self.name
        ext = ".png"
        archive_dir = os.path.abspath(self.archive_dir)
        fname_plot = os.path.join(archive_dir, fname + ext)
        os.makedirs(os.path.dirname(fname_plot), exist_ok=True)
        print(f"Saving plot to {fname_plot}")

        if reference_only:
            plt.figure(figsize=(8, 6))
            plt.plot(vza, brf)
            plt.xlabel("VZA [deg]")
            plt.ylabel("BRF in principal plane [-]")
            plt.title("Simulation result, can be used as new reference")
            plt.tight_layout()
            plt.savefig(fname_plot)
            plt.close()

        else:
            if "brf_srf" in self.value.data_vars:  # Handle spectral results
                brf_ref = np.squeeze(self.reference.brf_srf.values)
            else:  # Handle monochromatic results
                brf_ref = np.squeeze(self.reference.brf.values)

            regression_test_plots(
                brf_ref, brf, vza, fname_plot, (self.METRIC_NAME, metric_value)
            )


@parse_docs
@attrs.define
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


@parse_docs
@attrs.define
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
