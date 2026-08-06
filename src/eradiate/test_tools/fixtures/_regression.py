"""
Reference data management for regression tests, built on :mod:`pytest_regressions`.

This module owns everything that is *not* statistics: where reference datasets
are read from, where regenerated ones are written to, where artefacts land, and
how verdicts reach the test report. The comparison criteria themselves live in
:mod:`eradiate.test_tools.regression`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
import xarray as xr
from pytest_datadir.plugin import LazyDataDir
from pytest_regressions.common import perform_regression_check

from ..regression import (
    RegressionTest,
    RegressionTestFailure,
    RegressionTestOutcome,
)
from ..report import ReportLogger, figure_to_html, report_logger
from ... import fresolver
from ...config import SOURCE_DIR

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Location of regression test reference datasets, relative to a file resolver
#: search path.
REFERENCE_ROOT = "tests/regression_test_references"

#: File extension of reference datasets.
EXTENSION = ".nc"


@pytest.fixture(scope="session")
def reference_update_dir(pytestconfig, artefact_dir) -> Path:
    """
    Directory regenerated reference datasets are written to.

    This is deliberately *not* the directory references are read from. The file
    resolver may serve them from the asset manager's installation directory,
    whose entries are symbolic links into its unpack cache: writing through one
    would corrupt the cache. It resolves to, in order of precedence:

    * the ``--reference-dir`` option, if given;
    * the ``eradiate-data`` submodule working tree, in dev mode — a regenerated
      reference then shows up directly in ``git -C resources/data status``,
      ready to be reviewed and committed;
    * a subdirectory of the artefact directory otherwise.
    """
    option = pytestconfig.getoption("reference_dir")
    if option:
        return Path(option).resolve()

    if SOURCE_DIR:
        return (Path(SOURCE_DIR) / "resources/data" / REFERENCE_ROOT).resolve()

    return (Path(artefact_dir) / "references").resolve()


@pytest.fixture(scope="session")
def reference_dir(pytestconfig, reference_update_dir) -> Path:
    """
    Directory reference datasets are read from, resolved with the file resolver.

    Falls back to :func:`reference_update_dir` when the file resolver knows
    nothing about :data:`REFERENCE_ROOT`, so that a freshly regenerated
    reference is picked up by the next run.
    """
    option = pytestconfig.getoption("reference_dir")
    if option:
        return Path(option).resolve()

    resolved = fresolver.resolve(REFERENCE_ROOT)
    return resolved.resolve() if resolved.is_dir() else reference_update_dir


class DatasetRegressionFixture:
    """
    Implementation of the ``dataset_regression`` fixture: compare a simulation
    result against a stored reference dataset.

    Reference lookup, regeneration (``--force-regen``, ``--regen-all``) and
    artefact naming are delegated to
    `pytest-regressions <https://github.com/ESSS/pytest-regressions>`__; the
    verdict itself comes from one or several :class:`.RegressionTest` instances.
    """

    def __init__(
        self,
        request: pytest.FixtureRequest,
        tmp_path: Path,
        reference_dir: Path,
        reference_update_dir: Path,
        artefact_dir: Path,
        plot: bool,
        logger: ReportLogger = report_logger,
    ):
        self.request = request
        self.tmp_path = tmp_path
        self.reference_dir = reference_dir
        self.reference_update_dir = reference_update_dir
        self.artefact_dir = Path(artefact_dir)
        self.plot = plot
        self.logger = logger

    def reference_path(self, basename: str) -> Path | None:
        """
        Path of the reference dataset registered under `basename`, or ``None``
        if it does not exist.

        Useful to a test that must know whether a reference is available before
        deciding what to simulate, *e.g.* to render a high-sample-count
        reference with a different integrator.
        """
        path = self.reference_dir / f"{basename}{EXTENSION}"
        return path if path.is_file() else None

    def check(
        self,
        result: xr.Dataset,
        test: RegressionTest | Sequence[RegressionTest],
        basename: str,
    ) -> None:
        """
        Compare `result` against the reference dataset registered under
        `basename`.

        Parameters
        ----------
        result : Dataset
            Simulation result.

        test : :class:`.RegressionTest` or sequence of :class:`.RegressionTest`
            Criterion, or criteria, the comparison must satisfy. Several tests
            share a single reference dataset, which is how a case comparing
            more than one data variable is expressed.

        basename : str
            Name of the reference dataset within the reference directory,
            without its ``.nc`` extension. May contain forward slashes to
            address a subdirectory.

        Raises
        ------
        RegressionTestFailure
            If any of the criteria is not met.
        """
        __tracebackhide__ = True

        tests = [test] if isinstance(test, RegressionTest) else list(test)
        if not tests:
            raise ValueError("At least one regression test must be specified")

        self._guard_missing_reference(basename)

        # `dump_aux_fn` fires both when a reference has just been created and
        # when a comparison failed. Only the former has no reference to chart
        # against; in the latter case `check_fn` has already emitted the
        # comparison chart.
        bootstrapping = self.reference_path(basename) is None

        def dump_fn(filename: Path) -> None:
            filename.parent.mkdir(parents=True, exist_ok=True)
            result.to_netcdf(filename)

        def check_fn(obtained_filename: Path, expected_filename: Path) -> None:
            __tracebackhide__ = True
            reference = xr.load_dataset(expected_filename)
            self._compare(result, reference, tests, basename)

        def dump_aux_fn(filename: Path) -> list[str]:
            return self._plot_noref(result, tests[0], basename) if bootstrapping else []

        perform_regression_check(
            # Reads resolve against the (possibly read-only) reference
            # directory, writes go to the update directory. Keeping the two
            # apart is the whole point: see `reference_update_dir`.
            datadir=LazyDataDir(self.reference_dir, self.tmp_path),
            original_datadir=self.reference_update_dir,
            request=self.request,
            check_fn=check_fn,
            dump_fn=dump_fn,
            dump_aux_fn=dump_aux_fn,
            extension=EXTENSION,
            basename=basename,
            obtained_filename=self.artefact_dir / f"{basename}.obtained{EXTENSION}",
        )

    def _guard_missing_reference(self, basename: str) -> None:
        """
        Fail a test whose reference does not exist, unless reference creation
        was explicitly requested.

        Left to itself, :mod:`pytest_regressions` writes a missing reference and
        fails, which makes a typo in `basename` indistinguishable from a
        deliberate bootstrap.
        """
        __tracebackhide__ = True

        config = self.request.config
        if config.getoption("force_regen") or config.getoption("regen_all"):
            return

        if self.reference_path(basename) is None:
            pytest.fail(
                f"Regression test reference '{basename}{EXTENSION}' was not "
                f"found in '{self.reference_dir}'. If the reference is "
                "genuinely missing and should be created, re-run with the "
                "--force-regen flag; otherwise check the reference name."
            )

    def _compare(
        self,
        result: xr.Dataset,
        reference: xr.Dataset,
        tests: list[RegressionTest],
        basename: str,
    ) -> None:
        """
        Evaluate every criterion, report the outcomes, chart them, and raise if
        any of them failed.
        """
        __tracebackhide__ = True

        self.logger.info(f"Regression test {basename} results:")
        failures = []

        for test in tests:
            try:
                outcome = test.evaluate(result, reference)
            except Exception:
                # A malformed comparison is a bug, not a verdict. Chart the
                # offending data to help diagnose it, but never let a plotting
                # error replace the exception that matters.
                self.logger.info("An exception occurred during test evaluation!")
                try:
                    self._plot(result, reference, test, None, basename)
                except Exception as plot_error:
                    self.logger.info(
                        f"Could not plot the failed evaluation: {plot_error}"
                    )
                raise

            self.logger.info(str(outcome))
            for message in outcome.warnings:
                self.logger.warning(message)

            self._plot(result, reference, test, outcome, basename)

            if not outcome.passed:
                failures.append(
                    f"{outcome.metric_name} = {outcome.metric_value}, "
                    f"threshold = {outcome.threshold}, "
                    f"variable = '{outcome.variable}'"
                )

        if failures:
            raise RegressionTestFailure(
                f"Regression test '{basename}' did not pass: " + "; ".join(failures)
            )

    def _plot(
        self,
        result: xr.Dataset,
        reference: xr.Dataset,
        test: RegressionTest,
        outcome: RegressionTestOutcome | None,
        basename: str,
    ) -> None:
        """
        Chart a comparison, on success as well as on failure, so that a report
        run documents every test it ran.
        """
        if not self.plot:
            return

        fig, _ = test.plot(result, reference, outcome)
        try:
            # Several criteria may share one reference dataset, so the tested
            # variable disambiguates their charts.
            self._emit(fig, f"{basename}-{test.variable}.png")
        finally:
            plt.close(fig)

    def _plot_noref(
        self, result: xr.Dataset, test: RegressionTest, basename: str
    ) -> list[str]:
        """
        Chart a result that has just been archived as a reference candidate.
        """
        if not self.plot:
            return []

        fig, _ = test.plot_noref(result)
        try:
            return [str(self._emit(fig, f"{basename}.png"))]
        finally:
            plt.close(fig)

    def _emit(self, fig: Figure, filename: str) -> Path:
        """
        Send a figure to the test report and save it to the artefact directory.

        Artefacts never go to the reference directory: regenerating a reference
        must not leave PNG files behind in the data repository.
        """
        self.logger.html(figure_to_html(fig))

        path = self.artefact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

        # When a report backend is active the chart is embedded in the report
        # itself, which makes the PNG copy a secondary artefact not worth
        # announcing.
        if not self.logger.reporting:
            print(f"Saved plot to {path}")

        return path


@pytest.fixture
def dataset_regression(
    request, tmp_path, reference_dir, reference_update_dir, artefact_dir, plot_figures
) -> DatasetRegressionFixture:
    """
    Compare a simulation result against a stored reference dataset.

    Examples
    --------
    >>> def test_something(dataset_regression):
    ...     result = eradiate.run(exp)
    ...     dataset_regression.check(
    ...         result,
    ...         ZTest(threshold=0.05, variable="radiance"),
    ...         basename="something_ref",
    ...     )
    """
    return DatasetRegressionFixture(
        request=request,
        tmp_path=tmp_path,
        reference_dir=reference_dir,
        reference_update_dir=reference_update_dir,
        artefact_dir=artefact_dir,
        plot=plot_figures,
    )
