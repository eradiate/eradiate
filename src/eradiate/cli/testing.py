"""
Testing utilities for Eradiate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._console import console, message, section

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Testing utilities.
    """
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


@app.command(name="diff-metrics")
def diff_metrics(
    before: Annotated[
        Path,
        typer.Argument(
            help="Path to the baseline metrics report (JSON).",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    after: Annotated[
        Path,
        typer.Argument(
            help="Path to the metrics report to compare (JSON).",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show all differences instead of just significant ones.",
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Percentage threshold for significant duration changes (default: 10%).",
        ),
    ] = 10.0,
    min_duration: Annotated[
        float,
        typer.Option(
            "--min-duration",
            "-m",
            help="Minimum baseline duration (seconds) to report changes (default: 0.1s).",
        ),
    ] = 0.1,
):
    """
    Compare two test metrics reports and display differences.

    In concise mode (default), only significant changes are shown:
    environment changes, test regressions/fixes, and duration changes
    above the threshold for tests with baseline duration above --min-duration.

    In verbose mode (-v), all test duration changes are displayed.
    """
    from eradiate.test_tools.pytest_metrics import MetricReport, ReportComparison

    # Load reports
    try:
        report_before = MetricReport.load(before)
    except Exception as e:
        console.print(f"[red]Error loading {before}: {e}[/red]")
        raise typer.Exit(1)

    try:
        report_after = MetricReport.load(after)
    except Exception as e:
        console.print(f"[red]Error loading {after}: {e}[/red]")
        raise typer.Exit(1)

    # Compare
    comparison = ReportComparison.compare(report_before, report_after)

    if verbose:
        _print_verbose_comparison(comparison, threshold, min_duration)
    else:
        _print_concise_comparison(comparison, threshold, min_duration)


def _print_concise_comparison(comparison, threshold: float, min_duration: float):
    """Print only significant differences."""
    from eradiate.test_tools.pytest_metrics import ReportComparison

    comp: ReportComparison = comparison
    has_output = False

    # Environment changes
    if comp.environment.has_changes:
        section("Environment changes", newline=False)
        has_output = True

        if comp.environment.hostname_changed:
            message(
                f"  Hostname: {comp.environment.hostname_before} → "
                f"{comp.environment.hostname_after}"
            )
        if comp.environment.platform_changed:
            message(
                f"  Platform: {comp.environment.platform_before} → "
                f"{comp.environment.platform_after}"
            )
        if comp.environment.cpu_changed:
            message(
                f"  CPU: {comp.environment.cpu_before} → {comp.environment.cpu_after}"
            )
        if comp.environment.memory_changed:
            mem_before = comp.environment.memory_before_gb
            mem_after = comp.environment.memory_after_gb
            message(f"  Memory: {mem_before:.1f} GB → {mem_after:.1f} GB")
        if comp.environment.python_version_changed:
            message(
                f"  Python: {comp.environment.python_before} → "
                f"{comp.environment.python_after}"
            )
        if comp.environment.eradiate_version_changed:
            message(
                f"  Eradiate: {comp.environment.eradiate_before} → "
                f"{comp.environment.eradiate_after}"
            )
        if comp.environment.package_changes:
            message("  Packages:")
            for pkg, (v_before, v_after) in sorted(
                comp.environment.package_changes.items()
            ):
                message(f"    {pkg}: {v_before} → {v_after}")

    # Test count changes
    added = comp.added_tests
    removed = comp.removed_tests
    if added or removed:
        section("Test changes", newline=has_output)
        has_output = True

        if added:
            message(f"  Added: {len(added)} tests")
            for diff in added[:5]:
                message(f"    + {diff.node_id}")
            if len(added) > 5:
                message(f"    ... and {len(added) - 5} more")

        if removed:
            message(f"  Removed: {len(removed)} tests")
            for diff in removed[:5]:
                message(f"    - {diff.node_id}")
            if len(removed) > 5:
                message(f"    ... and {len(removed) - 5} more")

    # Outcome changes (regressions and fixes)
    regressions = comp.regressions
    fixes = comp.fixes
    if regressions or fixes:
        section("Outcome changes", newline=has_output)
        has_output = True

        if regressions:
            message(f"  Regressions (PASSED → FAILED): {len(regressions)}")
            for diff in regressions:
                message(f"    ✗ {diff.node_id}")

        if fixes:
            message(f"  Fixes (FAILED → PASSED): {len(fixes)}")
            for diff in fixes:
                message(f"    ✓ {diff.node_id}")

    # Significant duration changes (filtered by min_duration)
    slowest = [
        d
        for d in comp.slowest_regressions(threshold_percent=threshold, top_n=50)
        if d.duration_before is not None and d.duration_before >= min_duration
    ][:10]
    fastest = [
        d
        for d in comp.fastest_improvements(threshold_percent=threshold, top_n=50)
        if d.duration_before is not None and d.duration_before >= min_duration
    ][:10]

    if slowest or fastest:
        section(
            f"Duration changes (>{threshold:.0f}%, baseline ≥{min_duration:.2f}s)",
            newline=has_output,
        )
        has_output = True

        if slowest:
            message("  Slower:")
            for diff in slowest:
                pct = diff.duration_change_percent
                message(
                    f"    {diff.duration_before:.3f}s → {diff.duration_after:.3f}s "
                    f"({pct:+.1f}%) {diff.node_id}"
                )

        if fastest:
            message("  Faster:")
            for diff in fastest:
                pct = diff.duration_change_percent
                message(
                    f"    {diff.duration_before:.3f}s → {diff.duration_after:.3f}s "
                    f"({pct:+.1f}%) {diff.node_id}"
                )

    # Summary
    section("Summary", newline=has_output)
    message(f"  Tests: {comp.before.test_count} → {comp.after.test_count}")
    message(
        f"  Total duration: {comp.before.total_duration:.2f}s → "
        f"{comp.after.total_duration:.2f}s"
    )
    pct = comp.total_duration_change_percent
    if pct is not None:
        message(f"  Change: {comp.total_duration_change:+.2f}s ({pct:+.1f}%)")


def _print_verbose_comparison(comparison, threshold: float, min_duration: float):
    """Print all differences."""
    from eradiate.test_tools.pytest_metrics import ReportComparison

    comp: ReportComparison = comparison

    # Environment changes (same as concise)
    if comp.environment.has_changes:
        section("Environment changes", newline=False)

        if comp.environment.hostname_changed:
            message(
                f"  Hostname: {comp.environment.hostname_before} → "
                f"{comp.environment.hostname_after}"
            )
        if comp.environment.platform_changed:
            message(
                f"  Platform: {comp.environment.platform_before} → "
                f"{comp.environment.platform_after}"
            )
        if comp.environment.cpu_changed:
            message(
                f"  CPU: {comp.environment.cpu_before} → {comp.environment.cpu_after}"
            )
        if comp.environment.memory_changed:
            mem_before = comp.environment.memory_before_gb
            mem_after = comp.environment.memory_after_gb
            message(f"  Memory: {mem_before:.1f} GB → {mem_after:.1f} GB")
        if comp.environment.python_version_changed:
            message(
                f"  Python: {comp.environment.python_before} → "
                f"{comp.environment.python_after}"
            )
        if comp.environment.eradiate_version_changed:
            message(
                f"  Eradiate: {comp.environment.eradiate_before} → "
                f"{comp.environment.eradiate_after}"
            )
        if comp.environment.package_changes:
            message("  Packages:")
            for pkg, (v_before, v_after) in sorted(
                comp.environment.package_changes.items()
            ):
                message(f"    {pkg}: {v_before} → {v_after}")

    # Test changes
    added = comp.added_tests
    removed = comp.removed_tests
    if added or removed:
        section("Test changes")

        if added:
            message(f"  Added ({len(added)}):")
            for diff in added:
                message(f"    + {diff.node_id}")

        if removed:
            message(f"  Removed ({len(removed)}):")
            for diff in removed:
                message(f"    - {diff.node_id}")

    # Outcome changes
    regressions = comp.regressions
    fixes = comp.fixes
    if regressions or fixes:
        section("Outcome changes")

        if regressions:
            message(f"  Regressions ({len(regressions)}):")
            for diff in regressions:
                message(f"    ✗ {diff.node_id}")

        if fixes:
            message(f"  Fixes ({len(fixes)}):")
            for diff in fixes:
                message(f"    ✓ {diff.node_id}")

    # All duration changes (verbose shows everything)
    section("All duration changes")

    # Get all tests that exist in both reports, sorted by percentage change
    common_tests = [
        diff
        for diff in comp.test_diffs.values()
        if diff.status in ("changed", "unchanged")
        and diff.duration_change_percent is not None
    ]
    common_tests.sort(key=lambda d: d.duration_change_percent or 0, reverse=True)

    if common_tests:
        for diff in common_tests:
            pct = diff.duration_change_percent
            # Only mark as significant if above min_duration threshold
            marker = ""
            if (
                diff.duration_before is not None
                and diff.duration_before >= min_duration
            ):
                if pct > threshold:
                    marker = " [slower]"
                elif pct < -threshold:
                    marker = " [faster]"
            message(
                f"  {diff.duration_before:.3f}s → {diff.duration_after:.3f}s "
                f"({pct:+.1f}%){marker} {diff.node_id}"
            )
    else:
        message("  No common tests to compare.")

    # Summary
    section("Summary")
    message(f"  Tests: {comp.before.test_count} → {comp.after.test_count}")
    message(
        f"  Total duration: {comp.before.total_duration:.2f}s → "
        f"{comp.after.total_duration:.2f}s"
    )
    pct = comp.total_duration_change_percent
    if pct is not None:
        message(f"  Change: {comp.total_duration_change:+.2f}s ({pct:+.1f}%)")
