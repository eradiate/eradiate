"""
Pytest plugin for collecting and reporting test metrics.

This module provides a pytest plugin that records test duration and other
metrics, storing them in a JSON file for later analysis. It includes detailed
system metadata and supports comparison between test runs.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import attrs
import pytest

from .._version import _version as eradiate_version

# -----------------------------------------------------------------------------
#                              Data Models
# -----------------------------------------------------------------------------


def _convert_datetime(
    value: str | datetime.datetime | None,
) -> datetime.datetime | None:
    """Convert string or datetime to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


@attrs.define
class CPUInfo:
    """CPU hardware information."""

    model: str | None = None
    physical_cores: int | None = None
    logical_cores: int | None = None
    architecture: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CPUInfo:
        return cls(**data)

    @classmethod
    def collect(cls) -> CPUInfo:
        """Collect CPU information from the current system."""
        model = None
        physical_cores = None
        logical_cores = os.cpu_count()
        architecture = platform.machine()

        # Try to get CPU model
        if sys.platform == "linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            model = line.split(":", 1)[1].strip()
                            break
            except (OSError, IOError):
                pass

            # Get physical core count on Linux
            try:
                result = subprocess.run(
                    ["lscpu", "-p=Core,Socket"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Count unique core,socket pairs (excluding comments)
                cores = set()
                for line in result.stdout.strip().split("\n"):
                    if not line.startswith("#"):
                        cores.add(line)
                physical_cores = len(cores)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        elif sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                model = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                physical_cores = int(result.stdout.strip())
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass

        elif sys.platform == "win32":
            # Windows: use platform.processor() as fallback
            model = platform.processor() or None

        # Fallback for model
        if model is None:
            model = platform.processor() or None

        return cls(
            model=model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            architecture=architecture,
        )


@attrs.define
class MemoryInfo:
    """System memory information."""

    total_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryInfo:
        return cls(**data)

    @classmethod
    def collect(cls) -> MemoryInfo:
        """Collect memory information from the current system."""
        total_bytes = None

        if sys.platform == "linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Value is in kB
                            match = re.search(r"(\d+)", line)
                            if match:
                                total_bytes = int(match.group(1)) * 1024
                            break
            except (OSError, IOError):
                pass

        elif sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                total_bytes = int(result.stdout.strip())
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass

        elif sys.platform == "win32":
            try:
                result = subprocess.run(
                    [
                        "wmic",
                        "computersystem",
                        "get",
                        "TotalPhysicalMemory",
                        "/value",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in result.stdout.split("\n"):
                    if "TotalPhysicalMemory" in line:
                        total_bytes = int(line.split("=")[1].strip())
                        break
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass

        return cls(total_bytes=total_bytes)

    @property
    def total_gb(self) -> float | None:
        """Total memory in gigabytes."""
        if self.total_bytes is None:
            return None
        return self.total_bytes / (1024**3)


@attrs.define
class GPUInfo:
    """GPU hardware information for a single GPU."""

    model: str | None = None
    memory_bytes: int | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUInfo:
        return cls(**data)

    @property
    def memory_gb(self) -> float | None:
        """GPU memory in gigabytes."""
        if self.memory_bytes is None:
            return None
        return self.memory_bytes / (1024**3)


@attrs.define
class GPUList:
    """List of GPUs in the system."""

    gpus: list[GPUInfo] = attrs.field(factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUList:
        return cls(gpus=[GPUInfo.from_dict(g) for g in data.get("gpus", [])])

    @classmethod
    def collect(cls) -> GPUList:
        """Collect GPU information from the current system."""
        gpus: list[GPUInfo] = []

        # Try NVIDIA GPUs via nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        # Memory is in MiB
                        try:
                            memory_bytes = int(float(parts[1])) * 1024 * 1024
                        except ValueError:
                            memory_bytes = None
                        gpus.append(
                            GPUInfo(
                                model=parts[0],
                                memory_bytes=memory_bytes,
                                driver_version=parts[2],
                            )
                        )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return cls(gpus=gpus)


@attrs.define
class PythonEnvironment:
    """Python environment information."""

    version: str
    implementation: str
    executable: str
    packages: dict[str, str] = attrs.field(factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PythonEnvironment:
        return cls(**data)

    @classmethod
    def collect(cls, include_packages: list[str] | None = None) -> PythonEnvironment:
        """
        Collect Python environment information.

        Parameters
        ----------
        include_packages
            List of package names to include version info for.
            Defaults to a set of common scientific packages.
        """
        if include_packages is None:
            include_packages = [
                "eradiate",
                "mitsuba",
                "drjit",
                "numpy",
                "scipy",
                "xarray",
                "pandas",
                "attrs",
                "pint",
                "pytest",
            ]

        packages: dict[str, str] = {}
        for pkg_name in include_packages:
            try:
                from importlib.metadata import version

                packages[pkg_name] = version(pkg_name)
            except Exception:
                pass

        return cls(
            version=platform.python_version(),
            implementation=platform.python_implementation(),
            executable=sys.executable,
            packages=packages,
        )


@attrs.define
class SystemInfo:
    """Complete system information."""

    hostname: str
    platform: str
    platform_release: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: GPUList
    python: PythonEnvironment

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "platform_release": self.platform_release,
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "gpu": self.gpu.to_dict(),
            "python": self.python.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemInfo:
        return cls(
            hostname=data["hostname"],
            platform=data["platform"],
            platform_release=data["platform_release"],
            cpu=CPUInfo.from_dict(data["cpu"]),
            memory=MemoryInfo.from_dict(data["memory"]),
            gpu=GPUList.from_dict(data["gpu"]),
            python=PythonEnvironment.from_dict(data["python"]),
        )

    @classmethod
    def collect(cls) -> SystemInfo:
        """Collect system information from the current system."""
        return cls(
            hostname=platform.node(),
            platform=sys.platform,
            platform_release=platform.release(),
            cpu=CPUInfo.collect(),
            memory=MemoryInfo.collect(),
            gpu=GPUList.collect(),
            python=PythonEnvironment.collect(),
        )


@attrs.define
class GitInfo:
    """Git repository information."""

    commit_hash: str | None = None
    commit_time: str | None = None
    branch: str | None = None
    dirty: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitInfo:
        return cls(**data)

    @classmethod
    def collect(cls) -> GitInfo:
        """Collect git information from the current repository."""
        info = cls()

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            info.commit_hash = result.stdout.strip()

            result = subprocess.run(
                ["git", "log", "-1", "--format=%cI"],
                capture_output=True,
                text=True,
                check=True,
            )
            info.commit_time = result.stdout.strip()

            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            info.branch = result.stdout.strip()

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            info.dirty = bool(result.stdout.strip())

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return info


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
        default=None, converter=_convert_datetime
    )
    session_end: datetime.datetime | None = attrs.field(
        default=None, converter=_convert_datetime
    )

    # Metadata
    system: SystemInfo | None = None
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
            system=SystemInfo.from_dict(data["system"]) if data.get("system") else None,
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
class TestDiff:
    """Difference information for a single test."""

    node_id: str
    status: str  # "added", "removed", "changed", "unchanged"

    # Only for changed/unchanged tests
    duration_before: float | None = None
    duration_after: float | None = None
    outcome_before: str | None = None
    outcome_after: str | None = None

    @property
    def duration_change(self) -> float | None:
        """Absolute duration change in seconds."""
        if self.duration_before is None or self.duration_after is None:
            return None
        return self.duration_after - self.duration_before

    @property
    def duration_change_percent(self) -> float | None:
        """Relative duration change in percent."""
        if self.duration_before is None or self.duration_after is None:
            return None
        if self.duration_before == 0:
            return None
        return (
            (self.duration_after - self.duration_before) / self.duration_before
        ) * 100

    @property
    def outcome_changed(self) -> bool:
        """Whether the test outcome changed."""
        return self.outcome_before != self.outcome_after


@attrs.define
class EnvironmentDiff:
    """Differences in environment between two reports."""

    hostname_changed: bool = False
    hostname_before: str | None = None
    hostname_after: str | None = None

    platform_changed: bool = False
    platform_before: str | None = None
    platform_after: str | None = None

    cpu_changed: bool = False
    cpu_before: str | None = None
    cpu_after: str | None = None

    memory_changed: bool = False
    memory_before_gb: float | None = None
    memory_after_gb: float | None = None

    python_version_changed: bool = False
    python_before: str | None = None
    python_after: str | None = None

    eradiate_version_changed: bool = False
    eradiate_before: str | None = None
    eradiate_after: str | None = None

    package_changes: dict[str, tuple[str | None, str | None]] = attrs.field(
        factory=dict
    )

    @property
    def has_changes(self) -> bool:
        """Whether there are any environment changes."""
        return (
            self.hostname_changed
            or self.platform_changed
            or self.cpu_changed
            or self.memory_changed
            or self.python_version_changed
            or self.eradiate_version_changed
            or bool(self.package_changes)
        )


@attrs.define
class ReportComparison:
    """
    Comparison between two metric reports.

    This class provides detailed analysis of differences between two test runs,
    including environment changes and per-test performance differences.
    """

    before: MetricReport
    after: MetricReport
    environment: EnvironmentDiff
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
        env_diff = cls._compare_environment(before, after)
        test_diffs = cls._compare_tests(before, after)

        return cls(
            before=before,
            after=after,
            environment=env_diff,
            test_diffs=test_diffs,
        )

    @staticmethod
    def _compare_environment(
        before: MetricReport, after: MetricReport
    ) -> EnvironmentDiff:
        """Compare environment information between reports."""
        diff = EnvironmentDiff()

        # Compare hostnames
        hostname_before = before.system.hostname if before.system else None
        hostname_after = after.system.hostname if after.system else None
        if hostname_before != hostname_after:
            diff.hostname_changed = True
            diff.hostname_before = hostname_before
            diff.hostname_after = hostname_after

        # Compare platforms
        platform_before = before.system.platform if before.system else None
        platform_after = after.system.platform if after.system else None
        if platform_before != platform_after:
            diff.platform_changed = True
            diff.platform_before = platform_before
            diff.platform_after = platform_after

        # Compare CPUs
        cpu_before = (
            before.system.cpu.model if before.system and before.system.cpu else None
        )
        cpu_after = (
            after.system.cpu.model if after.system and after.system.cpu else None
        )
        if cpu_before != cpu_after:
            diff.cpu_changed = True
            diff.cpu_before = cpu_before
            diff.cpu_after = cpu_after

        # Compare memory
        mem_before = (
            before.system.memory.total_gb
            if before.system and before.system.memory
            else None
        )
        mem_after = (
            after.system.memory.total_gb
            if after.system and after.system.memory
            else None
        )
        if mem_before != mem_after:
            diff.memory_changed = True
            diff.memory_before_gb = mem_before
            diff.memory_after_gb = mem_after

        # Compare Python versions
        py_before = (
            before.system.python.version
            if before.system and before.system.python
            else None
        )
        py_after = (
            after.system.python.version
            if after.system and after.system.python
            else None
        )
        if py_before != py_after:
            diff.python_version_changed = True
            diff.python_before = py_before
            diff.python_after = py_after

        # Compare Eradiate versions
        if before.eradiate_version != after.eradiate_version:
            diff.eradiate_version_changed = True
            diff.eradiate_before = before.eradiate_version
            diff.eradiate_after = after.eradiate_version

        # Compare package versions
        pkgs_before = (
            before.system.python.packages
            if before.system and before.system.python
            else {}
        )
        pkgs_after = (
            after.system.python.packages if after.system and after.system.python else {}
        )
        all_packages = set(pkgs_before.keys()) | set(pkgs_after.keys())

        for pkg in all_packages:
            ver_before = pkgs_before.get(pkg)
            ver_after = pkgs_after.get(pkg)
            if ver_before != ver_after:
                diff.package_changes[pkg] = (ver_before, ver_after)

        return diff

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
                duration_after=test.duration,
                outcome_after=test.outcome,
            )

        # Tests removed in 'after'
        for node_id in before_ids - after_ids:
            test = before.tests[node_id]
            diffs[node_id] = TestDiff(
                node_id=node_id,
                status="removed",
                duration_before=test.duration,
                outcome_before=test.outcome,
            )

        # Tests present in both
        for node_id in before_ids & after_ids:
            test_before = before.tests[node_id]
            test_after = after.tests[node_id]

            outcome_changed = test_before.outcome != test_after.outcome
            status = "changed" if outcome_changed else "unchanged"

            diffs[node_id] = TestDiff(
                node_id=node_id,
                status=status,
                duration_before=test_before.duration,
                duration_after=test_after.duration,
                outcome_before=test_before.outcome,
                outcome_after=test_after.outcome,
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
            if d.outcome_before == "passed" and d.outcome_after == "failed"
        ]

    @property
    def fixes(self) -> list[TestDiff]:
        """Tests that went from failing to passing."""
        return [
            d
            for d in self.test_diffs.values()
            if d.outcome_before == "failed" and d.outcome_after == "passed"
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
            pct = diff.duration_change_percent
            if pct is not None and pct > threshold_percent:
                regressions.append(diff)

        regressions.sort(key=lambda d: d.duration_change_percent or 0, reverse=True)
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
            pct = diff.duration_change_percent
            if pct is not None and pct < -threshold_percent:
                improvements.append(diff)

        improvements.sort(key=lambda d: d.duration_change_percent or 0)
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
        if self.environment.has_changes:
            lines.append("Environment changes:")
            lines.append("-" * 40)

            if self.environment.hostname_changed:
                lines.append(
                    f"  Hostname: {self.environment.hostname_before} -> "
                    f"{self.environment.hostname_after}"
                )
            if self.environment.platform_changed:
                lines.append(
                    f"  Platform: {self.environment.platform_before} -> "
                    f"{self.environment.platform_after}"
                )
            if self.environment.cpu_changed:
                lines.append(
                    f"  CPU: {self.environment.cpu_before} -> "
                    f"{self.environment.cpu_after}"
                )
            if self.environment.memory_changed:
                lines.append(
                    f"  Memory: {self.environment.memory_before_gb:.1f} GB -> "
                    f"{self.environment.memory_after_gb:.1f} GB"
                )
            if self.environment.python_version_changed:
                lines.append(
                    f"  Python: {self.environment.python_before} -> "
                    f"{self.environment.python_after}"
                )
            if self.environment.eradiate_version_changed:
                lines.append(
                    f"  Eradiate: {self.environment.eradiate_before} -> "
                    f"{self.environment.eradiate_after}"
                )
            if self.environment.package_changes:
                lines.append("  Package changes:")
                for pkg, (v_before, v_after) in sorted(
                    self.environment.package_changes.items()
                ):
                    lines.append(f"    {pkg}: {v_before} -> {v_after}")
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
                pct = diff.duration_change_percent
                lines.append(
                    f"  {diff.node_id}\n"
                    f"    {diff.duration_before:.3f}s -> {diff.duration_after:.3f}s "
                    f"({pct:+.1f}%)"
                )
            lines.append("")

        # Performance improvements
        fastest = self.fastest_improvements(threshold_percent=20.0, top_n=5)
        if fastest:
            lines.append("Fastest improvements (>20% faster):")
            lines.append("-" * 40)
            for diff in fastest:
                pct = diff.duration_change_percent
                lines.append(
                    f"  {diff.node_id}\n"
                    f"    {diff.duration_before:.3f}s -> {diff.duration_after:.3f}s "
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
        self.report.system = SystemInfo.collect()
        self.report.git = GitInfo.collect()

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int):
        """Write metrics to file."""
        self.report.session_end = datetime.datetime.now(datetime.timezone.utc)

        if self.output_path is not None:
            self.report.save(self.output_path)
