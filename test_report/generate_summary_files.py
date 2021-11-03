import json
import os
import pathlib
import subprocess
import textwrap
from datetime import datetime

from tabulate import tabulate


def get_filename_testcasename(test):
    return test["nodeid"].split("::")


def get_git_revision_short_hash():
    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return rev.decode(encoding="UTF-8", errors="strict").strip("\n")


def group_results_unittests(report):
    passed = {}
    failed = {}
    skipped = {}
    for test in report["tests"]:
        [filename, testcasename] = get_filename_testcasename(test)
        if test["outcome"] == "passed":
            if filename in passed:
                passed[filename].append(testcasename)
            else:
                passed[filename] = [testcasename]
        elif test["outcome"] == "skipped":
            if filename in skipped:
                skipped[filename].append(testcasename)
            else:
                skipped[filename] = [testcasename]
        elif test["outcome"] == "failed":
            if filename in failed:
                failed[filename].append(testcasename)
            else:
                failed[filename] = [testcasename]
    return passed, failed, skipped


def create_summary_table(report):
    """
    Creates the summary section of the test report.
    This includes the test execution timestamp, the tested git revision and the
    number of passed, failed and skipped tests.
    """

    date = datetime.fromtimestamp(report["created"]).strftime("%Y-%m-%d")
    time = datetime.fromtimestamp(report["created"]).strftime("%H:%M:%S")
    commithash = get_git_revision_short_hash()

    heading = """.. _sec-summary:

*******
Summary
*******

This table contains the results of all tests that were executed in the creation 
of this test report. Additionally it contains the git revision that was used as 
well as the time of execution.
"""

    try:
        passed = report["summary"]["passed"]
    except KeyError:
        passed = 0
    try:
        failed = report["summary"]["failed"]
    except KeyError:
        failed = 0
    try:
        skipped = report["summary"]["skipped"]
    except KeyError:
        skipped = 0
    table = [
        ["Execution date", " ".join([date, time])],
        ["Git revision", commithash],
        [":ref:`Passed tests <Passed tests>`", passed],
        [":ref:`Failed tests <Failed tests>`", failed],
        [":ref:`Skipped tests <Skipped tests>`", skipped],
        ["Total", report["summary"]["total"]],
    ]
    tabulated = tabulate(table, tablefmt="rst")

    return "\n\n".join([heading, tabulated])


def create_report_list(test_data_dict, heading):
    """Create a list report for a set of tests, grouped by filename."""

    header = f"""
* {heading}"""

    body = ""
    for key, value in test_data_dict.items():
        value_joined = "\n    * ".join(value)
        body += f"""
  * {key}

    * {value_joined}
"""
    return "\n".join([header, body])


def create_report_dropdown(test_data_dict, heading, title_style="", body_style=""):
    """Create a dropdown report for a set of tests, grouped by filename."""

    n_tests = 0
    for k, v in test_data_dict.items():
        n_tests += len(v)

    header = [f".. dropdown:: {heading} [{n_tests}]"]

    if title_style:
        header.append(f"   :title: {title_style}")
    if body_style:
        header.append(f"   :body: {body_style}")

    header.append("\n")

    body = []
    for filename, test_names in test_data_dict.items():
        body_header = [f"   .. dropdown:: {filename} [{len(test_names)}]"]
        if title_style:
            body_header.append(f"      :title: {title_style}")
        if body_style:
            body_header.append(f"      :body: {body_style}")

        body.append("\n".join(body_header))
        body.append("\n".join(f"      * {test_name}" for test_name in test_names))

    return "\n".join(header) + "\n\n".join(body)


def generate():
    print("Parsing test results and generating overview files")
    eradiate_dir = pathlib.Path(os.environ["ERADIATE_DIR"])
    html_report_dir = eradiate_dir / "test_report"
    build_dir = html_report_dir / "generated"
    if not pathlib.Path.exists(build_dir):
        os.mkdir(build_dir)

    passed = dict()
    failed = dict()
    skipped = dict()

    for file in [
        f for f in os.listdir(build_dir) if os.path.isfile(os.path.join(build_dir, f))
    ]:
        if os.path.splitext(file)[-1] == ".json":
            with open(os.path.join(build_dir, file), "r") as json_file:
                report = json.load(json_file)
            passed_, failed_, skipped_ = group_results_unittests(report)
            passed = {**passed, **passed_}
            failed = {**failed, **failed_}
            skipped = {**skipped, **skipped_}

    with open(build_dir / "summary.rst", "w") as summary_file:
        summary_file.write(create_summary_table(report))

    for (
        test_data_dict,
        filename,
        section_heading,
        dropdown_title_style,
        dropdown_body_style,
    ) in [
        (
            passed,
            "passed.rst",
            "Passed tests",
            "bg-success text-light",
            "bg-success text-light",
        ),
        (
            failed,
            "failed.rst",
            "Failed tests",
            "bg-danger text-light",
            "bg-danger text-light",
        ),
        (
            skipped,
            "skipped.rst",
            "Skipped tests",
            "bg-warning text-dark",
            "bg-warning text-dark",
        ),
    ]:
        with open(build_dir / filename, "w") as f:
            f.write(f".. _{section_heading}:\n\n")

            # Report rendered as HTML dropdown
            f.write(".. only:: html\n\n")
            f.write(
                textwrap.indent(
                    create_report_dropdown(
                        test_data_dict,
                        section_heading,
                        title_style=dropdown_title_style,
                        body_style=dropdown_body_style,
                    ),
                    "   ",
                )
            )

            # Report rendered as list (for PDF version)
            f.write("\n\n")
            f.write(".. only:: not html\n\n")
            f.write(
                textwrap.indent(
                    create_report_list(test_data_dict, section_heading),
                    "   ",
                )
            )

            f.write("\n")


if __name__ == "__main__":
    generate()
