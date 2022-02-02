import os.path

from abc import ABC, abstractmethod
import attr
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ..typing import PathLike


def regression_test_plots(data: xr.Dataset, filename: PathLike, metric: tuple) -> None:
    """
    ...

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the reference and test result

    filename : path-like
        Path to the output file for the plot

    metric : tuple
        A tuple of the form (metric name, value) to be added to the plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=120)

    ref = np.squeeze(data.reference)
    result = np.squeeze(data.brf)
    vza = data.vza
    axes[0][0].plot(vza, ref, label="reference")
    axes[0][0].plot(vza, result, label="result")
    axes[0][0].set_title("Reference and test result")
    handles, labels = axes[0][0].get_legend_handles_labels()

    axes[1][0].plot(vza, result - ref)
    axes[1][0].set_xlabel("VZA in degrees")
    axes[1][0].set_ylabel("BRFpp")
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


@attr.s
class RenderTest(ABC):
    """
    Common interface for tests based on the comparison of a result array against
    reference values.
    """

    value: xr.DataArray = attr.ib()
    reference: xr.DataArray = attr.ib()
    threshold: float = attr.ib(default=0.05)
    archive_filename: PathLike = attr.ib(default=None)

    @abstractmethod
    def run(self) -> bool:
        """
        Execute the testing criterion and potentially create archive files.

        Depending on the specialized implementation, a metric will be computed
        from value and reference and compared to threshold.

        If archive_filename is set, a xarray.Dataset and a summary plot will be
        created. Otherwise, file creation will be omitted.

        Returns
        -------
        bool
            Result of the test criterion comparison.
        """
        pass


@attr.s
class RMSETest(RenderTest):
    """
    This class implements a simple test based on the root mean squared
    error (RMSE) of a result array with respect to the reference data.
    """

    def run(self) -> bool:
        value_np = self.value.brf.values
        ref_np = self.reference.brf.values
        if np.shape(value_np) != np.shape(self.reference):
            raise ValueError(
                f"Result and reference do not have the same shape!\n"
                f"Got: {np.shape(value_np)}, {np.shape(ref_np)}"
            )

        result_flat = np.array(value_np).flatten()
        ref_flat = np.array(ref_np).flatten()

        rmse = np.linalg.norm(result_flat - ref_flat) / np.sqrt(len(ref_flat))

        if self.archive_filename is not None:
            archive_dataset = self.reference.rename({"brf": "reference"})
            archive_dataset = archive_dataset.merge(self.value)
            archive_dirname = os.path.dirname(self.archive_filename)
            os.makedirs(archive_dirname, exist_ok=True)
            archive_dataset.to_netcdf(self.archive_filename)

            regression_test_plots(
                archive_dataset,
                "".join([self.archive_filename.splitext()[0], ".png"]),
                ("rmse", rmse),
            )

        return rmse <= self.threshold


@attr.s
class Chi2Test(RenderTest):
    """
    This class implements a statistical test for the regression testing
    campaign, based on Pearson's Chi-squared test.
    https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

    It determines the probability for the reference and the test result
    following the same distribution.

    """

    # The algorithm is adapted from mitsuba2's testing framework.

    def run(self) -> bool:
        from mitsuba.core.math import chi2
        from mitsuba.python.math import rlgamma

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

        chi2val, dof, pooled_in, pooled_out = chi2(
            histo_res_sorted, histo_ref_sorted, 5
        )

        p_value = 1.0 - rlgamma(dof / 2.0, chi2val / 2.0)

        if self.archive_filename is not None:
            archive_dataset = self.reference.rename({"brf": "reference"})
            archive_dataset = archive_dataset.merge(self.value, compat="override")
            archive_dirname = os.path.dirname(self.archive_filename)
            os.makedirs(archive_dirname, exist_ok=True)
            archive_dataset.to_netcdf(self.archive_filename)

            regression_test_plots(
                archive_dataset,
                "".join([os.path.splitext(self.archive_filename)[0], ".png"]),
                ("p-value", p_value),
            )

        if p_value < self.threshold:
            print(f"Failed the chi-squared test with p-value: {p_value}.")
            return False

        return True
