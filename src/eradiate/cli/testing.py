"""
Testing utilities for Eradiate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output comparison as JSON.",
        ),
    ] = False,
):
    """
    Compare two test metrics reports and display differences.

    In concise mode (default), only significant changes are shown:
    environment changes, test regressions/fixes, and duration changes
    above the threshold for tests with baseline duration above --min-duration.

    In verbose mode (-v), all test duration changes are displayed.

    Use --json to output the comparison as JSON for programmatic consumption.
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

    if json_output:
        _print_json_comparison(comparison, threshold, min_duration)
    elif verbose:
        _print_verbose_comparison(comparison, threshold, min_duration)
    else:
        _print_concise_comparison(comparison, threshold, min_duration)


def _print_json_comparison(comparison, threshold: float, min_duration: float):
    """Output comparison as JSON."""
    from eradiate.test_tools.pytest_metrics import ReportComparison

    comp: ReportComparison = comparison

    # Build environment changes dict
    env_changes: dict[str, Any] = {}
    for field_name, diff in comp.environment.fields.items():
        if diff.changed:
            env_changes[field_name] = {"before": diff.before, "after": diff.after}

    if comp.environment.package_changes:
        env_changes["packages"] = {
            pkg: {"before": diff.before, "after": diff.after}
            for pkg, diff in comp.environment.package_changes.items()
        }

    # Build test changes
    added_tests = [d.node_id for d in comp.added_tests]
    removed_tests = [d.node_id for d in comp.removed_tests]
    regressions = [d.node_id for d in comp.regressions]
    fixes = [d.node_id for d in comp.fixes]

    # Duration regressions
    slowest = [
        d
        for d in comp.slowest_regressions(threshold_percent=threshold, top_n=100)
        if d.duration.before is not None and d.duration.before >= min_duration
    ]
    duration_regressions = [
        {
            "node_id": d.node_id,
            "before": d.duration.before,
            "after": d.duration.after,
            "percent_change": d.duration.percent_change,
        }
        for d in slowest
    ]

    # Duration improvements
    fastest = [
        d
        for d in comp.fastest_improvements(threshold_percent=threshold, top_n=100)
        if d.duration.before is not None and d.duration.before >= min_duration
    ]
    duration_improvements = [
        {
            "node_id": d.node_id,
            "before": d.duration.before,
            "after": d.duration.after,
            "percent_change": d.duration.percent_change,
        }
        for d in fastest
    ]

    output = {
        "environment_changes": env_changes,
        "tests": {
            "before_count": comp.before.test_count,
            "after_count": comp.after.test_count,
            "added": added_tests,
            "removed": removed_tests,
            "regressions": regressions,
            "fixes": fixes,
        },
        "duration": {
            "before": comp.before.total_duration,
            "after": comp.after.total_duration,
            "change": comp.total_duration_change,
            "percent_change": comp.total_duration_change_percent,
            "regressions": duration_regressions,
            "improvements": duration_improvements,
        },
    }

    print(json.dumps(output, indent=2))


def _print_concise_comparison(comparison, threshold: float, min_duration: float):
    """Print only significant differences."""
    from eradiate.test_tools.pytest_metrics import ReportComparison

    comp: ReportComparison = comparison
    has_output = False

    # Environment changes
    if comp.environment.has_changes:
        section("Environment changes", newline=False)
        has_output = True

        # Display labels for each environment field
        field_labels = {
            "hostname": "Hostname",
            "platform": "Platform",
            "cpu": "CPU",
            "ram_gb": "RAM (GB)",
            "python": "Python",
            "eradiate_version": "Eradiate",
        }

        for field_name, label in field_labels.items():
            diff = comp.environment.fields.get(field_name)
            if diff and diff.changed:
                message(f"  {label}: {diff.before} -> {diff.after}")

        if comp.environment.package_changes:
            message("  Packages:")
            for pkg, pkg_diff in sorted(comp.environment.package_changes.items()):
                message(f"    {pkg}: {pkg_diff.before} -> {pkg_diff.after}")

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
            message(f"  Regressions (PASSED -> FAILED): {len(regressions)}")
            for diff in regressions:
                message(f"    x {diff.node_id}")

        if fixes:
            message(f"  Fixes (FAILED -> PASSED): {len(fixes)}")
            for diff in fixes:
                message(f"    v {diff.node_id}")

    # Significant duration changes (filtered by min_duration)
    slowest = [
        d
        for d in comp.slowest_regressions(threshold_percent=threshold, top_n=50)
        if d.duration.before is not None and d.duration.before >= min_duration
    ][:10]
    fastest = [
        d
        for d in comp.fastest_improvements(threshold_percent=threshold, top_n=50)
        if d.duration.before is not None and d.duration.before >= min_duration
    ][:10]

    if slowest or fastest:
        section(
            f"Duration changes (>{threshold:.0f}%, baseline >={min_duration:.2f}s)",
            newline=has_output,
        )
        has_output = True

        if slowest:
            message("  Slower:")
            for d in slowest:
                pct = d.duration.percent_change
                message(
                    f"    {d.duration.before:.3f}s -> {d.duration.after:.3f}s "
                    f"({pct:+.1f}%) {d.node_id}"
                )

        if fastest:
            message("  Faster:")
            for d in fastest:
                pct = d.duration.percent_change
                message(
                    f"    {d.duration.before:.3f}s -> {d.duration.after:.3f}s "
                    f"({pct:+.1f}%) {d.node_id}"
                )

    # Summary
    section("Summary", newline=has_output)
    message(f"  Tests: {comp.before.test_count} -> {comp.after.test_count}")
    message(
        f"  Total duration: {comp.before.total_duration:.2f}s -> "
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

        # Display labels for each environment field
        field_labels = {
            "hostname": "Hostname",
            "platform": "Platform",
            "cpu": "CPU",
            "ram_gb": "RAM (GB)",
            "python": "Python",
            "eradiate_version": "Eradiate",
        }

        for field_name, label in field_labels.items():
            diff = comp.environment.fields.get(field_name)
            if diff and diff.changed:
                message(f"  {label}: {diff.before} -> {diff.after}")

        if comp.environment.package_changes:
            message("  Packages:")
            for pkg, pkg_diff in sorted(comp.environment.package_changes.items()):
                message(f"    {pkg}: {pkg_diff.before} -> {pkg_diff.after}")

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
                message(f"    x {diff.node_id}")

        if fixes:
            message(f"  Fixes ({len(fixes)}):")
            for diff in fixes:
                message(f"    v {diff.node_id}")

    # All duration changes (verbose shows everything)
    section("All duration changes")

    # Get all tests that exist in both reports, sorted by percentage change
    common_tests = [
        diff
        for diff in comp.test_diffs.values()
        if diff.status in ("changed", "unchanged")
        and diff.duration.percent_change is not None
    ]
    common_tests.sort(key=lambda d: d.duration.percent_change or 0, reverse=True)

    if common_tests:
        for d in common_tests:
            pct = d.duration.percent_change
            # Only mark as significant if above min_duration threshold
            marker = ""
            if d.duration.before is not None and d.duration.before >= min_duration:
                if pct > threshold:
                    marker = " [slower]"
                elif pct < -threshold:
                    marker = " [faster]"
            message(
                f"  {d.duration.before:.3f}s -> {d.duration.after:.3f}s "
                f"({pct:+.1f}%){marker} {d.node_id}"
            )
    else:
        message("  No common tests to compare.")

    # Summary
    section("Summary")
    message(f"  Tests: {comp.before.test_count} -> {comp.after.test_count}")
    message(
        f"  Total duration: {comp.before.total_duration:.2f}s -> "
        f"{comp.after.total_duration:.2f}s"
    )
    pct = comp.total_duration_change_percent
    if pct is not None:
        message(f"  Change: {comp.total_duration_change:+.2f}s ({pct:+.1f}%)")
