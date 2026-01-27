"""
Pytest plugin for collecting and reporting test metrics.

This module provides a pytest plugin that records test duration and other
metrics, storing them in a JSON file for later analysis. It includes system
metadata and supports comparison between test runs.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import attrs
import pytest

from .._version import _version as eradiate_version
from ..util.sys_info import GitInfo, SysInfo

# -----------------------------------------------------------------------------
#                              Data Models
# -----------------------------------------------------------------------------


def _convert_datetime(value: str | datetime.datetime) -> datetime.datetime | None:
    """Convert string or datetime to datetime."""
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


@attrs.define
class TestResult:
    """Result for a single test."""

    node_id: str
    duration: float  # seconds
    outcome: str  # passed, failed, skipped, xfailed, xpassed
    markers: list[str] = attrs.field(factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 6),
            "outcome": self.outcome,
            "markers": self.markers if self.markers else None,
        }

    @classmethod
    def from_dict(cls, node_id: str, data: dict[str, Any]) -> TestResult:
        return cls(
            node_id=node_id,
            duration=data["duration"],
            outcome=data["outcome"],
            markers=data.get("markers") or [],
        )


@attrs.define
class MetricReport:
    """
    Complete test metrics report.

    This class represents a complete snapshot of a test run, including
    system information, git state, and individual test results.
    """

    # Report format version for future compatibility
    version: int = 1

    # Session timing
    session_start: datetime.datetime | None = attrs.field(
        default=None, converter=attrs.converters.optional(_convert_datetime)
    )
    session_end: datetime.datetime | None = attrs.field(
        default=None, converter=attrs.converters.optional(_convert_datetime)
    )

    # Metadata
    system: SysInfo | None = None
    git: GitInfo | None = None
    eradiate_version: str | None = None

    # Test results
    tests: dict[str, TestResult] = attrs.field(factory=dict)

    @property
    def total_duration(self) -> float:
        """Total duration of all tests in seconds."""
        return sum(t.duration for t in self.tests.values())

    @property
    def test_count(self) -> int:
        """Number of tests in the report."""
        return len(self.tests)

    @property
    def passed_count(self) -> int:
        """Number of passed tests."""
        return sum(1 for t in self.tests.values() if t.outcome == "passed")

    @property
    def failed_count(self) -> int:
        """Number of failed tests."""
        return sum(1 for t in self.tests.values() if t.outcome == "failed")

    @property
    def skipped_count(self) -> int:
        """Number of skipped tests."""
        return sum(1 for t in self.tests.values() if t.outcome == "skipped")

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a dictionary suitable for JSON serialization."""
        return {
            "version": self.version,
            "session_start": (
                self.session_start.isoformat() if self.session_start else None
            ),
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "total_duration": self.total_duration,
            "test_count": self.test_count,
            "system": self.system.to_dict() if self.system else None,
            "git": self.git.to_dict() if self.git else None,
            "eradiate_version": self.eradiate_version,
            "tests": {
                node_id: t.to_dict() for node_id, t in sorted(self.tests.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricReport:
        """Create a report from a dictionary."""
        tests = {}
        for node_id, test_data in data.get("tests", {}).items():
            tests[node_id] = TestResult.from_dict(node_id, test_data)

        return cls(
            version=data.get("version", 1),
            session_start=data.get("session_start"),
            session_end=data.get("session_end"),
            system=SysInfo.from_dict(data["system"]) if data.get("system") else None,
            git=GitInfo.from_dict(data["git"]) if data.get("git") else None,
            eradiate_version=data.get("eradiate_version"),
            tests=tests,
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> MetricReport:
        """Deserialize report from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path | str) -> None:
        """Save report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> MetricReport:
        """Load report from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


# -----------------------------------------------------------------------------
#                           Report Comparison
# -----------------------------------------------------------------------------


@attrs.define
class ItemDiff:
    """
    Generic difference container for before/after comparison of a single item.

    This class captures the state of an item in two reports and provides
    utilities to determine if the item has changed.
    """

    before: Any = None
    after: Any = None

    @property
    def changed(self) -> bool:
        """Whether the value changed between before and after."""
        return self.before != self.after

    @property
    def change(self) -> tuple[Any, Any] | None:
        """Return (before, after) tuple if changed, None otherwise."""
        if self.changed:
            return self.before, self.after
        return None


@attrs.define
class NumericDiff(ItemDiff):
    """
    Difference container for numeric values with change calculation support.

    Extends ItemDiff with methods for calculating absolute and percentage changes.
    """

    @property
    def absolute_change(self) -> float | None:
        """Absolute change (after - before)."""
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def percent_change(self) -> float | None:
        """Percentage change relative to before value."""
        if self.before is None or self.after is None:
            return None
        if self.before == 0:
            return None
        return ((self.after - self.before) / self.before) * 100


@attrs.define
class TestDiff:
    """Difference information for a single test."""

    node_id: str
    status: str  # "added", "removed", "changed", "unchanged"
    duration: NumericDiff = attrs.field(factory=NumericDiff)
    outcome: ItemDiff = attrs.field(factory=ItemDiff)


# SysInfo fields to compare: maps field name to display label
SYSINFO_FIELDS: dict[str, str] = {
    "hostname": "Hostname",
    "platform": "Platform",
    "platform_release": "Platform Release",
    "os": "OS",
    "cpu": "CPU",
    "ram_gb": "RAM (GB)",
    "llvm_version": "LLVM",
    "gpus": "GPUs",
    "cuda_version": "CUDA",
    "python": "Python",
    "drjit_version": "Dr.Jit",
    "mitsuba_version": "Mitsuba",
    "eradiate_mitsuba_version": "Eradiate-Mitsuba",
    "mitsuba_compiler": "Mitsuba Compiler",
}


@attrs.define
class SysInfoDiff:
    """
    Differences between two SysInfo instances.

    Uses a dictionary mapping field names to ItemDiff instances for scalar fields,
    and a separate dict for package version changes.
    """

    before: SysInfo | None = None
    after: SysInfo | None = None
    fields: dict[str, ItemDiff] = attrs.field(factory=dict)
    package_changes: dict[str, ItemDiff] = attrs.field(factory=dict)

    @classmethod
    def compare(cls, before: SysInfo | None, after: SysInfo | None) -> SysInfoDiff:
        """
        Create a SysInfoDiff by comparing two SysInfo instances.

        Parameters
        ----------
        before
            The baseline SysInfo.
        after
            The SysInfo to compare against the baseline.

        Returns
        -------
        SysInfoDiff
            Difference between the two SysInfo instances.
        """
        fields = {}
        for field_name in SYSINFO_FIELDS:
            val_before = getattr(before, field_name, None) if before else None
            val_after = getattr(after, field_name, None) if after else None
            fields[field_name] = ItemDiff(before=val_before, after=val_after)

        # Compare package versions
        pkgs_before = before.packages if before else {}
        pkgs_after = after.packages if after else {}
        all_packages = set(pkgs_before.keys()) | set(pkgs_after.keys())

        package_changes = {}
        for pkg in all_packages:
            pkg_diff = ItemDiff(before=pkgs_before.get(pkg), after=pkgs_after.get(pkg))
            if pkg_diff.changed:
                package_changes[pkg] = pkg_diff

        return cls(
            before=before,
            after=after,
            fields=fields,
            package_changes=package_changes,
        )

    @property
    def has_changes(self) -> bool:
        """Whether there are any SysInfo changes."""
        return any(diff.changed for diff in self.fields.values()) or bool(
            self.package_changes
        )


@attrs.define
class ReportComparison:
    """
    Comparison between two metric reports.

    This class provides detailed analysis of differences between two test runs,
    including system info changes and per-test performance differences.
    """

    before: MetricReport
    after: MetricReport
    sysinfo: SysInfoDiff
    eradiate_version: ItemDiff = attrs.field(factory=ItemDiff)
    test_diffs: dict[str, TestDiff] = attrs.field(factory=dict)

    @classmethod
    def compare(cls, before: MetricReport, after: MetricReport) -> ReportComparison:
        """
        Compare two metric reports.

        Parameters
        ----------
        before
            The baseline report (e.g., previous run).
        after
            The report to compare against the baseline (e.g., current run).

        Returns
        -------
        ReportComparison
            Detailed comparison results.
        """
        sysinfo_diff = SysInfoDiff.compare(before.system, after.system)
        eradiate_version_diff = ItemDiff(
            before=before.eradiate_version, after=after.eradiate_version
        )
        test_diffs = cls._compare_tests(before, after)

        return cls(
            before=before,
            after=after,
            sysinfo=sysinfo_diff,
            eradiate_version=eradiate_version_diff,
            test_diffs=test_diffs,
        )

    @staticmethod
    def _compare_tests(
        before: MetricReport, after: MetricReport
    ) -> dict[str, TestDiff]:
        """Compare test results between reports."""
        diffs: dict[str, TestDiff] = {}

        before_ids = set(before.tests.keys())
        after_ids = set(after.tests.keys())

        # Tests added in 'after'
        for node_id in after_ids - before_ids:
            test = after.tests[node_id]
            diffs[node_id] = TestDiff(
                node_id=node_id,
                status="added",
                duration=NumericDiff(after=test.duration),
                outcome=ItemDiff(after=test.outcome),
            )

        # Tests removed in 'after'
        for node_id in before_ids - after_ids:
            test = before.tests[node_id]
            diffs[node_id] = TestDiff(
                node_id=node_id,
                status="removed",
                duration=NumericDiff(before=test.duration),
                outcome=ItemDiff(before=test.outcome),
            )

        # Tests present in both
        for node_id in before_ids & after_ids:
            test_before = before.tests[node_id]
            test_after = after.tests[node_id]

            outcome = ItemDiff(before=test_before.outcome, after=test_after.outcome)
            status = "changed" if outcome.changed else "unchanged"

            diffs[node_id] = TestDiff(
                node_id=node_id,
                status=status,
                duration=NumericDiff(
                    before=test_before.duration, after=test_after.duration
                ),
                outcome=outcome,
            )

        return diffs

    @property
    def added_tests(self) -> list[TestDiff]:
        """Tests added in the 'after' report."""
        return [d for d in self.test_diffs.values() if d.status == "added"]

    @property
    def removed_tests(self) -> list[TestDiff]:
        """Tests removed from the 'before' report."""
        return [d for d in self.test_diffs.values() if d.status == "removed"]

    @property
    def outcome_changes(self) -> list[TestDiff]:
        """Tests with changed outcomes."""
        return [d for d in self.test_diffs.values() if d.status == "changed"]

    @property
    def regressions(self) -> list[TestDiff]:
        """Tests that went from passing to failing."""
        return [
            d
            for d in self.test_diffs.values()
            if d.outcome.before == "passed" and d.outcome.after == "failed"
        ]

    @property
    def fixes(self) -> list[TestDiff]:
        """Tests that went from failing to passing."""
        return [
            d
            for d in self.test_diffs.values()
            if d.outcome.before == "failed" and d.outcome.after == "passed"
        ]

    def slowest_regressions(
        self, threshold_percent: float = 10.0, top_n: int = 10
    ) -> list[TestDiff]:
        """
        Get tests with the largest performance regressions.

        Parameters
        ----------
        threshold_percent
            Minimum percentage increase to be considered a regression.
        top_n
            Maximum number of regressions to return.

        Returns
        -------
        list[TestDiff]
            Tests with duration increases above the threshold, sorted by
            percentage change (largest first).
        """
        regressions = []
        for diff in self.test_diffs.values():
            if diff.status not in ("changed", "unchanged"):
                continue
            pct = diff.duration.percent_change
            if pct is not None and pct > threshold_percent:
                regressions.append(diff)

        regressions.sort(key=lambda d: d.duration.percent_change or 0, reverse=True)
        return regressions[:top_n]

    def fastest_improvements(
        self, threshold_percent: float = 10.0, top_n: int = 10
    ) -> list[TestDiff]:
        """
        Get tests with the largest performance improvements.

        Parameters
        ----------
        threshold_percent
            Minimum percentage decrease to be considered an improvement.
        top_n
            Maximum number of improvements to return.

        Returns
        -------
        list[TestDiff]
            Tests with duration decreases above the threshold, sorted by
            percentage change (largest improvement first).
        """
        improvements = []
        for diff in self.test_diffs.values():
            if diff.status not in ("changed", "unchanged"):
                continue
            pct = diff.duration.percent_change
            if pct is not None and pct < -threshold_percent:
                improvements.append(diff)

        improvements.sort(key=lambda d: d.duration.percent_change or 0)
        return improvements[:top_n]

    @property
    def total_duration_change(self) -> float:
        """Total change in test duration in seconds."""
        return self.after.total_duration - self.before.total_duration

    @property
    def total_duration_change_percent(self) -> float | None:
        """Total change in test duration as a percentage."""
        if self.before.total_duration == 0:
            return None
        return (self.total_duration_change / self.before.total_duration) * 100

    def summary(self) -> str:
        """Generate a human-readable summary of the comparison."""
        lines = ["=" * 60, "Test metrics comparison summary", "=" * 60, ""]

        # Environment changes
        has_env_changes = self.sysinfo.has_changes or self.eradiate_version.changed
        if has_env_changes:
            lines.append("Environment changes:")
            lines.append("-" * 40)

            # Eradiate version (from MetricReport, not SysInfo)
            if self.eradiate_version.changed:
                lines.append(
                    f"  Eradiate: {self.eradiate_version.before} -> "
                    f"{self.eradiate_version.after}"
                )

            # SysInfo fields
            for field_name, label in SYSINFO_FIELDS.items():
                diff = self.sysinfo.fields.get(field_name)
                if diff and diff.changed:
                    lines.append(f"  {label}: {diff.before} -> {diff.after}")

            if self.sysinfo.package_changes:
                lines.append("  Package changes:")
                for pkg, pkg_diff in sorted(self.sysinfo.package_changes.items()):
                    lines.append(f"    {pkg}: {pkg_diff.before} -> {pkg_diff.after}")
            lines.append("")

        # Test count changes
        lines.append("Test Counts:")
        lines.append("-" * 40)
        lines.append(f"  Before: {self.before.test_count} tests")
        lines.append(f"  After:  {self.after.test_count} tests")
        lines.append(f"  Added:  {len(self.added_tests)}")
        lines.append(f"  Removed: {len(self.removed_tests)}")
        lines.append("")

        # Outcome changes
        if self.regressions or self.fixes:
            lines.append("Outcome changes:")
            lines.append("-" * 40)
            lines.append(f"  Regressions (pass -> fail): {len(self.regressions)}")
            lines.append(f"  Fixes (fail -> pass): {len(self.fixes)}")

            if self.regressions:
                lines.append("\n  Regressions:")
                for diff in self.regressions[:5]:
                    lines.append(f"    - {diff.node_id}")
                if len(self.regressions) > 5:
                    lines.append(f"    ... and {len(self.regressions) - 5} more")

            if self.fixes:
                lines.append("\n  Fixes:")
                for diff in self.fixes[:5]:
                    lines.append(f"    - {diff.node_id}")
                if len(self.fixes) > 5:
                    lines.append(f"    ... and {len(self.fixes) - 5} more")
            lines.append("")

        # Duration changes
        lines.append("Duration:")
        lines.append("-" * 40)
        lines.append(f"  Before: {self.before.total_duration:.2f}s")
        lines.append(f"  After:  {self.after.total_duration:.2f}s")
        pct = self.total_duration_change_percent
        pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
        lines.append(f"  Change: {self.total_duration_change:+.2f}s{pct_str}")
        lines.append("")

        # Performance regressions
        slowest = self.slowest_regressions(threshold_percent=20.0, top_n=5)
        if slowest:
            lines.append("Slowest regressions (>20% slower):")
            lines.append("-" * 40)
            for diff in slowest:
                pct = diff.duration.percent_change
                lines.append(
                    f"  {diff.node_id}\n"
                    f"    {diff.duration.before:.3f}s -> {diff.duration.after:.3f}s "
                    f"({pct:+.1f}%)"
                )
            lines.append("")

        # Performance improvements
        fastest = self.fastest_improvements(threshold_percent=20.0, top_n=5)
        if fastest:
            lines.append("Fastest improvements (>20% faster):")
            lines.append("-" * 40)
            for diff in fastest:
                pct = diff.duration.percent_change
                lines.append(
                    f"  {diff.node_id}\n"
                    f"    {diff.duration.before:.3f}s -> {diff.duration.after:.3f}s "
                    f"({pct:+.1f}%)"
                )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# -----------------------------------------------------------------------------
#                              Pytest Plugin
# -----------------------------------------------------------------------------


class TestMetricsPlugin:
    """
    Pytest plugin to collect and report test metrics.

    This plugin records test duration and other metrics, optionally storing them
    in a JSON file for later analysis. It collects detailed system information
    and supports comparison between test runs.

    Test durations are reported inline as tests complete only in verbose mode.

    Parameters
    ----------
    output_path : Path or None
        Path to the JSON file where metrics will be written.
    """

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path
        self.report = MetricReport(eradiate_version=eradiate_version)
        self._verbose: bool = False
        self._terminal: pytest.TerminalReporter | None = None

    def pytest_configure(self, config: pytest.Config):
        """Store config for verbosity check."""
        self._verbose = config.option.verbose > 0
        self._terminal = config.pluginmanager.get_plugin("terminalreporter")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo):
        """Capture test duration and outcome after each test."""
        outcome = yield
        report = outcome.get_result()

        # Only record metrics for the "call" phase (actual test execution)
        if report.when == "call":
            node_id = item.nodeid
            markers = [m.name for m in item.iter_markers()]

            self.report.tests[node_id] = TestResult(
                node_id=node_id,
                duration=report.duration,
                outcome=report.outcome,
                markers=markers,
            )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_report_teststatus(
        self, report: pytest.TestReport, config: pytest.Config
    ):
        """Append duration to test status in verbose mode."""
        outcome = yield

        if report.when != "call" or not self._verbose:
            return

        result = outcome.get_result()
        if result is None:
            return

        category, letter, word = result

        # Append duration to the word shown in verbose mode
        if isinstance(word, str):
            word_with_duration = f"{word} ({report.duration:.3f}s)"
        else:
            # word can be a tuple (word, markup_dict)
            word_with_duration = (f"{word[0]} ({report.duration:.3f}s)", word[1])

        outcome.force_result((category, letter, word_with_duration))

    def pytest_sessionstart(self, session: pytest.Session):
        """Record session start time and collect system info."""
        self.report.session_start = datetime.datetime.now(datetime.timezone.utc)
        self.report.system = SysInfo.collect()
        self.report.git = GitInfo.collect()

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int):
        """Write metrics to file."""
        self.report.session_end = datetime.datetime.now(datetime.timezone.utc)

        if self.output_path is not None:
            self.report.save(self.output_path)
