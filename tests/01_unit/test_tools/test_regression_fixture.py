"""
Tests for the ``dataset_regression`` fixture, i.e. for reference data
management. The statistical criteria themselves are tested in
``test_regression.py``.

The fixture is driven directly rather than through ``pytester``: the flags it
reads (``--force-regen``, ``--regen-all``) belong to :mod:`pytest_regressions`
and are covered upstream, whereas what needs pinning here is Eradiate's own
behaviour — the missing-reference guard, the separation between the directory
references are read from and the one they are written to, and where charts and
archives end up.
"""

import numpy as np
import pytest
import xarray as xr

from eradiate.test_tools.fixtures._regression import DatasetRegressionFixture
from eradiate.test_tools.regression import (
    RegressionTestFailure,
    RMSETest,
    ZTest,
)
from eradiate.test_tools.report import ReportLogger


class ReportLoggerSpy(ReportLogger):
    """
    Report logger that records messages and HTML fragments for assertions while
    forwarding them to the active report backend. Content sent through this spy
    therefore shows up in the generated test report and can be inspected
    visually.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []
        self.warnings = []
        self.fragments = []

    def info(self, msg):
        self.messages.append(msg)
        super().info(msg)

    def warning(self, msg):
        self.warnings.append(msg)
        super().warning(msg)

    def html(self, fragment):
        self.fragments.append(fragment)
        super().html(fragment)


class FakeRequest:
    """
    Minimal stand-in for a Pytest request. ``perform_regression_check`` only
    ever queries the three options below when a basename is given explicitly,
    which the fixture always does.
    """

    def __init__(self, **options):
        defaults = {
            "force_regen": False,
            "regen_all": False,
            "with_test_class_names": False,
        }
        self.config = type(
            "FakeConfig",
            (),
            {"getoption": lambda _self, name: {**defaults, **options}[name]},
        )()


def make_dataset(offset=0.0, n=8):
    vza = np.linspace(0.0, 60.0, n)
    da = xr.DataArray(np.ones(n) + offset, dims="vza", coords={"vza": vza})
    return xr.Dataset({"brf": da, "brf_var": xr.full_like(da, 1e-4)})


@pytest.fixture
def read_dir(tmp_path):
    """Directory references are read from."""
    path = tmp_path / "read"
    path.mkdir()
    return path


@pytest.fixture
def write_dir(tmp_path):
    """Directory regenerated references are written to."""
    return tmp_path / "write"


@pytest.fixture
def artefact_dir(tmp_path):
    """Directory charts and archived results are written to."""
    return tmp_path / "artefacts"


@pytest.fixture
def make_fixture(tmp_path, read_dir, write_dir, artefact_dir):
    """
    Build a :class:`.DatasetRegressionFixture` over throwaway directories.
    Reads and writes are deliberately pointed at *different* directories, as
    they are in production when the file resolver serves references from the
    asset manager.
    """

    def _make(plot=False, **options):
        return DatasetRegressionFixture(
            request=FakeRequest(**options),
            tmp_path=tmp_path / "work",
            reference_dir=read_dir,
            reference_update_dir=write_dir,
            artefact_dir=artefact_dir,
            plot=plot,
            logger=ReportLoggerSpy(use_robot=False),
        )

    return _make


def store_reference(read_dir, basename, dataset):
    """Seed the read-only reference directory."""
    path = read_dir / f"{basename}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


class TestMissingReference:
    """
    A missing reference is a setup error unless reference creation was
    explicitly requested: a typo in the reference name must not be mistaken for
    a deliberate bootstrap.
    """

    def test_without_flag(self, make_fixture, write_dir, artefact_dir):
        fixture = make_fixture()

        with pytest.raises(pytest.fail.Exception, match="--force-regen"):
            fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "missing")

        # nothing was written anywhere
        assert not write_dir.exists()
        assert not artefact_dir.exists()

    def test_with_force_regen(self, make_fixture, write_dir):
        # the bootstrap writes the candidate reference and still fails, so that
        # it can never be mistaken for a passing run
        fixture = make_fixture(force_regen=True)

        with pytest.raises(pytest.fail.Exception, match="File not found"):
            fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "bootstrap")

        assert (write_dir / "bootstrap.nc").is_file()

    def test_bootstrap_chart(self, make_fixture, artefact_dir):
        # with plotting on, the candidate is charted so that it can be vetted
        # before promotion
        fixture = make_fixture(plot=True, force_regen=True)

        with pytest.raises(pytest.fail.Exception):
            fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "bootstrap")

        assert (artefact_dir / "bootstrap.png").is_file()
        assert len(fixture.logger.fragments) == 1

    def test_reference_path(self, make_fixture, read_dir):
        # the probe a test uses to decide what to simulate when no reference is
        # available yet
        fixture = make_fixture()
        assert fixture.reference_path("probe") is None

        store_reference(read_dir, "probe", make_dataset())
        assert fixture.reference_path("probe") == read_dir / "probe.nc"


class TestComparison:
    """
    These tests check the comparison itself: verdicts, reporting and artefacts.
    """

    def test_pass(self, make_fixture, read_dir, artefact_dir):
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture()

        fixture.check(make_dataset(0.1), RMSETest(0.25, variable="brf"), "case")

        assert any("Test passed" in msg for msg in fixture.logger.messages)
        # the result is archived for inspection, the reference is not touched
        assert (artefact_dir / "case.obtained.nc").is_file()

    def test_fail(self, make_fixture, read_dir):
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture()

        with pytest.raises(RegressionTestFailure) as exc_info:
            fixture.check(make_dataset(3.0), RMSETest(0.25, variable="brf"), "case")

        # the numbers that decided the verdict must be in the message: a plain
        # pytest run does not show the report log
        message = str(exc_info.value)
        assert "case" in message
        assert "rmse" in message
        assert "0.25" in message

    def test_failure_leaves_reference_untouched(self, make_fixture, read_dir):
        # without a regeneration flag, a failing comparison must not rewrite
        # the reference
        path = store_reference(read_dir, "case", make_dataset())
        before = path.read_bytes()
        fixture = make_fixture()

        with pytest.raises(RegressionTestFailure):
            fixture.check(make_dataset(3.0), RMSETest(0.25, variable="brf"), "case")

        assert path.read_bytes() == before

    def test_reads_and_writes_are_separate(self, make_fixture, read_dir, write_dir):
        # the crux of the design: references are read through the file
        # resolver, which may serve them from the asset manager's installation
        # directory (a tree of symbolic links into its unpack cache), so
        # regeneration must never write back to where it read from
        path = store_reference(read_dir, "case", make_dataset())
        before = path.read_bytes()
        fixture = make_fixture(force_regen=True)

        with pytest.raises(pytest.fail.Exception, match="Files differ"):
            fixture.check(make_dataset(3.0), RMSETest(0.25, variable="brf"), "case")

        assert path.read_bytes() == before
        assert (write_dir / "case.nc").is_file()
        assert xr.load_dataset(write_dir / "case.nc")["brf"].values[0] == 4.0

    def test_subdirectory_basename(self, make_fixture, read_dir):
        # references addressed through a subdirectory, as the RAMI4ATM cases are
        store_reference(read_dir, "group/case-ref", make_dataset())
        fixture = make_fixture()

        fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "group/case-ref")

    def test_malformed_data_raises_through(self, make_fixture, read_dir):
        # a broken comparison must surface as itself, not as a test verdict and
        # not as a plotting error
        store_reference(read_dir, "case", make_dataset(n=4))
        fixture = make_fixture(plot=True)

        with pytest.raises(ValueError, match="do not have the same shape"):
            fixture.check(make_dataset(n=8), RMSETest(0.25, variable="brf"), "case")


class TestMultipleTests:
    """
    Several criteria may share a single reference dataset, which is how a case
    comparing more than one data variable is expressed.
    """

    @staticmethod
    def make_two_variable_dataset(offset=0.0):
        ds = make_dataset()
        ds["hdrf"] = ds["brf"] + offset
        return ds

    def test_all_evaluated(self, make_fixture, read_dir, artefact_dir):
        store_reference(read_dir, "case", self.make_two_variable_dataset())
        fixture = make_fixture(plot=True)

        fixture.check(
            self.make_two_variable_dataset(),
            [RMSETest(0.25, variable="brf"), RMSETest(0.25, variable="hdrf")],
            "case",
        )

        # one chart per criterion, named after the variable it tested
        assert (artefact_dir / "case-brf.png").is_file()
        assert (artefact_dir / "case-hdrf.png").is_file()

    def test_reports_every_failure(self, make_fixture, read_dir):
        store_reference(read_dir, "case", self.make_two_variable_dataset())
        fixture = make_fixture()

        with pytest.raises(RegressionTestFailure) as exc_info:
            fixture.check(
                self.make_two_variable_dataset(offset=3.0),
                [RMSETest(0.25, variable="brf"), RMSETest(0.25, variable="hdrf")],
                "case",
            )

        # brf is unchanged, hdrf is off: the message must name the one that failed
        assert "'hdrf'" in str(exc_info.value)
        assert "'brf'" not in str(exc_info.value)

    def test_rejects_empty(self, make_fixture):
        fixture = make_fixture()

        with pytest.raises(ValueError, match="At least one regression test"):
            fixture.check(make_dataset(), [], "case")


class TestReporting:
    """
    These tests check plot gating and report routing, which the fixture owns.
    """

    def test_plot_gating(self, make_fixture, read_dir, artefact_dir):
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture(plot=False)

        fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "case")

        assert not fixture.logger.fragments
        assert not artefact_dir.exists() or not list(artefact_dir.glob("*.png"))

    def test_chart_on_success(self, make_fixture, read_dir, artefact_dir):
        # charts are produced on pass as well as on failure: a report run must
        # document every test it ran
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture(plot=True)

        fixture.check(make_dataset(), RMSETest(0.25, variable="brf"), "case")

        assert (artefact_dir / "case-brf.png").is_file()
        assert len(fixture.logger.fragments) == 1
        assert fixture.logger.fragments[0].startswith("<svg")

    def test_chart_on_failure(self, make_fixture, read_dir, artefact_dir):
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture(plot=True)

        with pytest.raises(RegressionTestFailure):
            fixture.check(make_dataset(3.0), RMSETest(0.25, variable="brf"), "case")

        assert (artefact_dir / "case-brf.png").is_file()

    def test_warnings_are_surfaced(self, make_fixture, read_dir):
        # a Z-test reference without variance degrades the test; the fixture
        # must relay the warning the criterion produced
        reference = make_dataset().drop_vars("brf_var")
        store_reference(read_dir, "case", reference)
        fixture = make_fixture()

        fixture.check(make_dataset(), ZTest(0.05, variable="brf"), "case")

        assert any("brf_var" in message for message in fixture.logger.warnings)

    def test_no_artefacts_in_reference_dir(self, make_fixture, read_dir, write_dir):
        # regenerating a reference must not leave charts behind in the data
        # repository
        store_reference(read_dir, "case", make_dataset())
        fixture = make_fixture(plot=True, force_regen=True)

        with pytest.raises(pytest.fail.Exception):
            fixture.check(make_dataset(3.0), RMSETest(0.25, variable="brf"), "case")

        assert not list(read_dir.glob("*.png"))
        assert not list(write_dir.glob("*.png"))
