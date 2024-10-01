"""
Release utility.
"""

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

import click
import tomlkit
from packaging.version import Version


def _run(cmd, cwd=".", show_output=False):
    print(f"  Running: '{' '.join(cmd)}' in '{cwd}'")
    out = (
        subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE)
        .stdout.strip()
        .decode("utf-8")
    )

    if show_output:
        print(f"  Output: {out}")

    return out


def get_package_requirements(pyproject_path):
    with open(pyproject_path, "r") as f:
        pyproject = tomlkit.load(f)

    version_optional = Version(
        [
            x
            for x in pyproject["project"]["optional-dependencies"]["optional"]
            if x.startswith("eradiate-mitsuba")
        ][0].split("==")[1]
    )

    return version_optional


def get_mi_header_versions(header_path):
    with open(header_path, "r") as f:
        contents = f.read()

    mi_version = re.search(
        r"#define\ MI\_VERSION\_MAJOR\ (?P<major>\d+)\n"
        r"#define\ MI\_VERSION\_MINOR\ (?P<minor>\d+)\n"
        r"#define\ MI\_VERSION\_PATCH\ (?P<patch>\d+)",
        contents,
    ).groupdict()
    erd_mi_version = re.search(
        r"#define\ ERD\_MI\_VERSION\_MAJOR\ (?P<major>\d+)\n"
        r"#define\ ERD\_MI\_VERSION\_MINOR\ (?P<minor>\d+)\n"
        r"#define\ ERD\_MI\_VERSION\_PATCH\ (?P<patch>\d+)",
        contents,
    ).groupdict()
    result = {
        "mi_version": Version(
            f'{mi_version["major"]}.{mi_version["minor"]}.{mi_version["patch"]}',
        ),
        "erd_mi_version": Version(
            f'{erd_mi_version["major"]}.{erd_mi_version["minor"]}.{erd_mi_version["patch"]}',
        ),
    }
    return result


def get_kernel_requirements(module_path):
    with open(module_path, "r") as f:
        contents = f.read()

    mi_version = re.search(
        r"REQUIRED\_MITSUBA\_VERSION\ =\ "
        r"\"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\"",
        contents,
    ).groupdict()
    erd_mi_version = re.search(
        r"REQUIRED\_MITSUBA\_PATCH\_VERSION\ =\ "
        r"\"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\"",
        contents,
    ).groupdict()

    result = {
        "mitsuba_version": Version(
            f'{mi_version["major"]}.{mi_version["minor"]}.{mi_version["patch"]}',
        ),
        "mitsuba_patch_version": Version(
            f'{erd_mi_version["major"]}.{erd_mi_version["minor"]}.{erd_mi_version["patch"]}',
        ),
    }
    return result


def get_git_describe(cwd):
    cmd = ["git", "describe", "--dirty", "--tags", "--long", "--match", "v[0-9]*"]
    s = _run(cmd, cwd)
    sections = s.split("-")
    result = {
        "version": Version(sections[0]),
        "commits": int(sections[1]),
        "hash": sections[2][1:],
    }
    return result


def get_git_submodule_status():
    cmd = ["git", "submodule", "status"]
    s = _run(cmd)
    sections = s.split("\n")
    mi_submodule = [x for x in sections if "ext/mitsuba" in x][0].strip()
    result = {
        "hash_mismatch": mi_submodule.startswith("+"),
        "hash": mi_submodule.strip("+").split()[0][:9],
    }
    return result


def _check_mitsuba_requirements(
    eradiate_root_dir: Path, ignore_submodule_not_on_tag: bool = False
) -> int:
    """
    Check if Mitsuba requirements are consistent with checked out version.
    """
    info = {}
    mitsuba_root_dir = eradiate_root_dir / "ext/mitsuba"

    # Get required Mitsuba version
    print("Looking up required eradiat-mitsuba package version ...")
    info["required_mitsuba_package_optional"] = get_package_requirements(
        "pyproject.toml"
    )

    # Get checked out Mitsuba version
    print("Looking up checked out Mitsuba Git commit ...")
    describe = get_git_describe(mitsuba_root_dir)
    info["submodule_latest_tag_version"] = describe["version"]
    info["submodule_commits_since_tag"] = describe["commits"]

    # Get submodule state
    print("Getting mitsuba submodule state ...")
    submodule_status = get_git_submodule_status()
    info["submodule_hash_mismatch"] = submodule_status["hash_mismatch"]
    info["submodule_current_short_hash"] = submodule_status["hash"]

    # Get Mitsuba header versions
    print("Getting Mitsuba header versions ...")
    mi_header_versions = get_mi_header_versions(
        mitsuba_root_dir / "include/mitsuba/mitsuba.h"
    )
    info["mi_header_mitsuba_version"] = mi_header_versions["mi_version"]
    info["mi_header_mitsuba_patch_version"] = mi_header_versions["erd_mi_version"]

    # Get Eradiate kernel requirements
    print("Getting Eradiate kernel requirements ...")
    eradiate_kernel_versions = get_kernel_requirements(
        eradiate_root_dir / "src/eradiate/kernel/_versions.py"
    )
    info["eradiate_kernel_mitsuba_version"] = eradiate_kernel_versions[
        "mitsuba_version"
    ]
    info["eradiate_kernel_mitsuba_patch_version"] = eradiate_kernel_versions[
        "mitsuba_patch_version"
    ]

    # Display diagnostics and how to fix issues
    diagnostics = {}

    mitsuba_versions = {
        info["required_mitsuba_package_optional"],
        info["eradiate_kernel_mitsuba_patch_version"],
        info["mi_header_mitsuba_patch_version"],
        info["submodule_latest_tag_version"],
    }
    if len(mitsuba_versions) > 1:
        diagnostics["mitsuba_version_mismatch"] = (
            "The following versions are not aligned:\n"
            f"* eradiate-mitsuba package requirement [{info['required_mitsuba_package_optional']}]\n"
            "  retrieved from 'pyproject.toml' ('optional' group)\n"
            f"* mitsuba submodule patch version [{info['mi_header_mitsuba_patch_version']}]\n"
            "  retrieved from 'mitsuba.h'\n"
            f"* mitsuba patch requirement [{info['eradiate_kernel_mitsuba_patch_version']}]\n"
            "  retrieved from 'eradiate.kernel._versions'\n"
            f"* latest mitsuba submodule tag [{info['submodule_latest_tag_version']}]\n"
            "  retrieved with the 'git describe' command"
        )

    mitsuba_upstream_versions = {
        info["eradiate_kernel_mitsuba_version"],
        info["mi_header_mitsuba_version"],
    }
    if len(mitsuba_upstream_versions) > 1:
        diagnostics["mitsuba_upstream_version_mismatch"] = (
            "The following versions are not aligned:\n"
            f"* mitsuba upstream version [{info['mi_header_mitsuba_version']}]\n"
            "  retrieved from 'mitsuba.h'\n"
            f"* mitsuba upstream requirement [{info['eradiate_kernel_mitsuba_version']}]\n"
            "  retrieved from 'eradiate.kernel._versions'"
        )

    if info["submodule_hash_mismatch"] is True:
        diagnostics["git_submodule_hash_mismatch"] = (
            "Checked out mitsuba submodule commit does not match the SHA-1 "
            "found in the index of the eradiate repository.\n"
            "Update the submodule before releasing."
        )

    if info["submodule_commits_since_tag"] > 0 and not ignore_submodule_not_on_tag:
        diagnostics["submodule_not_on_tag"] = (
            "Checked out mitsuba submodule commit does not match the latest tag.\n"
            "This likely means that you are trying to release an Eradiate "
            "version tested with an unreleased version of the kernel."
        )

    print()
    if diagnostics:
        messages = [diag for diag in diagnostics.values()]
        print("\n\n".join(messages))
        return 1
    else:
        print("No issues detected")
        return 0


@click.group()
def cli():
    """
    Eradiate release utility.

    This tool performs release operations for the Eradiate software package.
    """
    pass


@cli.command()
@click.option(
    "-l",
    "--eradiate-root-dir",
    default=".",
    help="Path to the root of the Eradiate codebase. Default: '.'",
)
@click.option(
    "--ignore-submodule-not-on-tag",
    is_flag=True,
)
def check_mitsuba(eradiate_root_dir, ignore_submodule_not_on_tag):
    """
    Check if Mitsuba requirements are consistent.
    """

    code = _check_mitsuba_requirements(
        Path(eradiate_root_dir), ignore_submodule_not_on_tag
    )
    sys.exit(code)


@cli.command()
def update_citation():
    """
    Update CITATION.cff file.
    """
    target_version = Version(os.environ["RELEASE_VERSION"])

    citation_file = open("CITATION.cff", "r").read()
    print(citation_file)

    # Replace version
    m = re.search(
        r"^version: (?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$",
        citation_file,
        flags=re.M,
    )
    if m:
        old_version = Version(f"{m[1]}.{m[2]}.{m[3]}")
        new_version = target_version
        print(f"Replacing old version ({old_version}) with new version ({new_version})")
        citation_file = citation_file.replace(m[0], f"version: {new_version}")
    else:
        raise RuntimeError("could not find release version line in CITATION.cff")

    # Replace date
    m = re.search(r"date-released:\ '(\d{4}-\d{2}-\d{2})'", citation_file)
    if m:
        old_date = m[1]
        new_date = datetime.datetime.today()
        print(f"Replacing old date ({old_date}) with new date ({new_date:%Y-%m-%d})")
        citation_file = citation_file.replace(
            m[0], f"date-released: '{new_date:%Y-%m-%d}'"
        )
    else:
        raise RuntimeError("could not find release date line in CITATION.cff")

    with open("CITATION.cff", "w") as f:
        f.write(citation_file)


if __name__ == "__main__":
    cli()
