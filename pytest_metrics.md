# pytest-metrics

This document describes an update to Eradiate that ultimately aims to implement metric reports in pytest, as well as a means to compare two reports.

## Requirements

### Report content

Reported information includes:

* System information
  * Hostname
  * Platform
  * CPU (model name only)
  * RAM (in GB, as a string)
  * GPU list (model names and amount of RAM as a string)
  * Python version
  * Dr.Jit and Mitsuba versions, available Mitsuba variants
  * Selected installed Python package versions (list should be flexible)
* Git state (optional, when in a repository)
  * Commit hash
  * Branch name
  * Dirty status
  * Commit timestamp
* Session metadata
  * Start and end timestamps
  * Eradiate version
* Per-test results
  * Node ID
  * Markers
  * Outcome
  * Duration
* Aggregated metrics (computed from per-test results)
  * Total duration
  * Test counts by outcome (passed, failed, skipped)

### Report format

* Export to JSON
* Include schema version for forward compatibility

### Pytest plugin

* Adds a `--metrics` flag that displays metrics (currently only duration) for each test when in verbose mode
* Adds a `--metrics-output` flag that implies `--metrics` and exports a `MetricReport` JSON dump

### CLI comparison

* Invoked as `eradiate testing diff-metrics BEFORE AFTER`
* Two reporting modes:
  * Concise (default): shows only changed entries
  * Verbose (`-v` flag): shows all entries before and after, with changes highlighted
* Thresholds for filtering insignificant changes:
  * `--min-duration`: minimum test duration to consider
  * `--threshold`: relative change threshold for numeric metrics
* Output format:
  * Colored terminal output with clear visual indicators for changes
  * Plain text fallback for non-TTY output
  * Optional JSON output (`--json`) for machine-readable diff results

### Report format compatibility

* Gradual report format upgrades should be supported (one version at a time), with each time the opportunity to warn the user about incompatibilities and fallback strategies

### Error handling

* Malformed JSON: report clear error with file path and parsing issue
* Incompatible report versions: warn and attempt best-effort comparison
* Partial system info collection failure: continue with available data, log warnings

## Data model

### Reporting classes (in `pytest_metrics`)

* `TestResult`: per-test metrics (node_id, duration, outcome, markers)
* `MetricReport`: complete session snapshot
  * Contains: version, session timestamps, system info, git info, Eradiate version, test results
  * Provides: aggregated metrics as computed properties

### Diff classes (in `pytest_metrics`)

Generic containers:

* `ItemDiff`: generic before/after container with `changed` property
* `NumericDiff`: extends `ItemDiff` with `absolute_change` and `percent_change`

Specialized comparisons:

* `TestDiff`: per-test comparison (status: added/removed/changed/unchanged, duration diff, outcome diff)
* `EnvironmentDiff`: system and package version comparison
* `ReportComparison`: orchestrates full report comparison, provides filtered views (regressions, fixes, slowest/fastest changes)

### System info classes (in `util.sys_info`)

* `SysInfo`: system information container with `collect()` class method
* `GitInfo`: git repository state with `collect()` class method

## Implementation guidelines

* Use attrs (`@attrs.define`) to declare data classes.
* Implement a `DictMixin` class providing `to_dict()` and `from_dict()` for attrs classes; override in subclasses only when custom serialization is needed.
* Implement system information collection as a `SysInfo.collect()` class method.
* Implement git information collection as a `GitInfo.collect()` class method.
* Reuse algorithms implemented by `util.sys_info` where possible.
* Handle missing optional data gracefully (git info when not in a repo, GPU info when unavailable).
