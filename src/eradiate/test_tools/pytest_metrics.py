"""
Pytest plugin for collecting and reporting test metrics.

This module provides a pytest plugin that records test duration and other
metrics, storing them in a YAML file for later analysis. It is designed to
eventually replace asv-based benchmarking with a simpler, home-grown solution.
"""

from __future__ import annotations

import datetime
import platform
import subprocess
from pathlib import Path
from typing import Any

import attrs
import pytest
import yaml

from .._version import _version as eradiate_version


@attrs.define
class TestMetrics:
    """Container for a single test's metrics."""

    node_id: str
    duration: float  # seconds
    outcome: str  # passed, failed, skipped, xfailed, xpassed
    markers: list[str] = attrs.field(factory=list)


class TestMetricsPlugin:
    """
    Pytest plugin to collect and report test metrics.

    This plugin records test duration and other metrics, storing them in a
    YAML file for later analysis. It is designed to eventually replace
    asv-based benchmarking with a simpler, home-grown solution.

    Parameters
    ----------
    output_path : Path or None
        Path to the YAML file where metrics will be written.
    top_n : int
        Number of slowest tests to display in the terminal summary.
    """

    def __init__(self, output_path: Path | None = None, top_n: int = 10):
        self.output_path = output_path
        self.top_n = top_n
        self.metrics: dict[str, TestMetrics] = {}
        self.session_start: datetime.datetime | None = None
        self.session_end: datetime.datetime | None = None
        self._collection_duration: float = 0.0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo):
        """Capture test duration and outcome after each test."""
        outcome = yield
        report = outcome.get_result()

        # Only record metrics for the "call" phase (actual test execution)
        if report.when == "call":
            node_id = item.nodeid
            markers = [m.name for m in item.iter_markers()]

            self.metrics[node_id] = TestMetrics(
                node_id=node_id,
                duration=report.duration,
                outcome=report.outcome,
                markers=markers,
            )

    def pytest_sessionstart(self, session: pytest.Session):
        """Record session start time."""
        self.session_start = datetime.datetime.now(datetime.timezone.utc)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int):
        """Write metrics to file and display summary."""
        self.session_end = datetime.datetime.now(datetime.timezone.utc)

        if self.output_path is not None:
            self._write_report()

    def pytest_terminal_summary(
        self, terminalreporter: pytest.TerminalReporter, exitstatus: int
    ):
        """Display slowest tests in terminal output."""
        if not self.metrics:
            return

        terminalreporter.write_sep("=", "slowest tests")

        # Sort by duration descending
        sorted_metrics = sorted(
            self.metrics.values(), key=lambda m: m.duration, reverse=True
        )

        for metrics in sorted_metrics[: self.top_n]:
            terminalreporter.write_line(
                f"{metrics.duration:>8.3f}s {metrics.outcome:<7} {metrics.node_id}"
            )

        # Total duration
        total_duration = sum(m.duration for m in self.metrics.values())
        terminalreporter.write_line(f"\nTotal test duration: {total_duration:.3f}s")

    def _get_git_info(self) -> dict[str, Any]:
        """Retrieve git repository information."""
        info: dict[str, Any] = {}

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["commit_hash"] = result.stdout.strip()

            # Get commit timestamp
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cI"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["commit_time"] = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["branch"] = result.stdout.strip()

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["dirty"] = bool(result.stdout.strip())

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return info

    def _get_environment_info(self) -> dict[str, Any]:
        """Retrieve environment information."""
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "eradiate_version": eradiate_version,
        }

    def _write_report(self):
        """Write metrics report to YAML file."""
        if self.output_path is None:
            return

        # Build report structure
        report: dict[str, Any] = {
            "metadata": {
                "session_start": (
                    self.session_start.isoformat() if self.session_start else None
                ),
                "session_end": (
                    self.session_end.isoformat() if self.session_end else None
                ),
                "total_duration": sum(m.duration for m in self.metrics.values()),
                "test_count": len(self.metrics),
                "git": self._get_git_info(),
                "environment": self._get_environment_info(),
            },
            "tests": {},
        }

        # Add test metrics
        for node_id, metric in sorted(self.metrics.items()):
            report["tests"][node_id] = {
                "duration": round(metric.duration, 6),
                "outcome": metric.outcome,
                "markers": metric.markers if metric.markers else None,
            }

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML with explicit options for readability
        with open(self.output_path, "w") as f:
            yaml.dump(
                report,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120,
            )
