#!/usr/bin/env python3
"""
Check that the eradiate-mitsuba wheel build matrix is a subset of Dr.Jit's.

eradiate-mitsuba wheels pin an exact Dr.Jit version, so every (platform,
Python) pair we build must also have a Dr.Jit wheel. This script compares:

* the ``os`` x ``python`` matrix of ``.github/workflows/wheels.yml``;
* the cibuildwheel ``build`` list in ``pyproject.toml``;

of ``ext/mitsuba`` against the same files of ``ext/mitsuba/ext/drjit``.
Runner labels are compared as platforms (OS and architecture), because the
two projects may pin different runner images for the same platform.

It also checks that the ``ext/drjit`` checkout matches the Dr.Jit version
pinned in ``pyproject.toml``, since the comparison is only meaningful against
the Dr.Jit release we depend on.

Run from the ``ext/mitsuba`` repository root (requires Python >= 3.11 and
PyYAML)::

    python check_ci_matrix.py

Exit status is 1 if any check fails.
"""

import re
import subprocess
import sys
from pathlib import Path

import tomllib
import yaml

DRJIT_DIR = Path("ext/drjit")
WORKFLOW = Path(".github/workflows/wheels.yml")


def runner_platform(label: str) -> str:
    """Map a GitHub runner label to an ``<os>-<arch>`` platform string."""
    if label.startswith("ubuntu-"):
        return "linux-arm64" if label.endswith("-arm") else "linux-x86_64"
    if label.startswith("windows-"):
        return "windows-arm64" if label.endswith("-arm") else "windows-x86_64"
    if label.startswith("macos-"):
        # Intel images are tagged explicitly; macos-13 is the last untagged one
        if label.endswith("-intel") or label in ("macos-12", "macos-13"):
            return "macos-x86_64"
        return "macos-arm64"
    raise ValueError(f"Unknown runner label {label!r}: extend runner_platform()")


def matrix(root: Path) -> tuple[set[str], set[str]]:
    wf = yaml.safe_load((root / WORKFLOW).read_text())
    m = wf["jobs"]["build_wheels"]["strategy"]["matrix"]
    return {runner_platform(os) for os in m["os"]}, set(m["python"])


def pyproject(root: Path) -> dict:
    return tomllib.loads((root / "pyproject.toml").read_text())


def main() -> int:
    errors = []

    # The Dr.Jit checkout must be the pinned release
    pin = next(
        d
        for d in pyproject(Path("."))["project"]["dependencies"]
        if d.startswith("drjit==")
    ).split("==")[1]
    tag = subprocess.run(
        ["git", "-C", str(DRJIT_DIR), "describe", "--tags", "--exact-match"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tag.lstrip("v") != pin:
        errors.append(
            f"ext/drjit is at {tag or 'an untagged commit'}, pyproject pins "
            f"drjit=={pin}: run 'git submodule update ext/drjit' first"
        )

    # Workflow matrix
    our_plat, our_py = matrix(Path("."))
    dr_plat, dr_py = matrix(DRJIT_DIR)
    if extra := our_plat - dr_plat:
        errors.append(f"platforms without Dr.Jit wheels: {sorted(extra)}")
    if extra := our_py - dr_py:
        errors.append(f"Python builds without Dr.Jit wheels: {sorted(extra)}")

    # cibuildwheel build selectors
    our_build = set(pyproject(Path("."))["tool"]["cibuildwheel"]["build"])
    dr_build = set(pyproject(DRJIT_DIR)["tool"]["cibuildwheel"]["build"])
    if extra := our_build - dr_build:
        errors.append(f"cibuildwheel builds without Dr.Jit wheels: {sorted(extra)}")

    # Our build list must agree with our own matrix (cp312_stable -> cp312)
    matrix_cp = {p.split("_")[0] for p in our_py}
    build_cp = {re.sub(r"-\*$", "", b) for b in our_build}
    if matrix_cp != build_cp:
        errors.append(
            f"workflow matrix Python {sorted(matrix_cp)} differs from "
            f"pyproject build list {sorted(build_cp)}"
        )

    print(f"Dr.Jit {pin}")
    print(f"  platforms: ours {sorted(our_plat)}, Dr.Jit {sorted(dr_plat)}")
    print(f"  python:    ours {sorted(our_py)}, Dr.Jit {sorted(dr_py)}")
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    if not errors:
        print("OK: build matrix is a subset of Dr.Jit's")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
