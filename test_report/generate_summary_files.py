import json
from datetime import datetime
import subprocess
from tabulate import tabulate
import os
import pathlib


def get_filename_testcasename(test):
    return test['nodeid'].split("::")


def get_git_revision_short_hash():
    rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return rev.decode(encoding='UTF-8', errors='strict').strip("\n")


def group_results_unittests(report):
    passed = {}
    failed = {}
    skipped = {}
    for test in report['tests']:
        [filename, testcasename] = get_filename_testcasename(test)
        if test['outcome'] == 'passed':
            if filename in passed:
                passed[filename].append(testcasename)
            else:
                passed[filename] = [testcasename]
        elif test['outcome'] == 'skipped':
            if filename in skipped:
                skipped[filename].append(testcasename)
            else:
                skipped[filename] = [testcasename]
        elif test['outcome'] == 'failed':
            if filename in failed:
                failed[filename].append(testcasename)
            else:
                failed[filename] = [testcasename]
    return passed, failed, skipped


def create_summary_table(report):
    """Creates the summary section of the test report.
    This includes the test execution timestamp, the tested git revision and the number of passed, failed and skipped tests."""

    date = datetime.fromtimestamp(report['created']).strftime('%Y-%m-%d')
    time = datetime.fromtimestamp(report['created']).strftime('%H:%M:%S')
    commithash = get_git_revision_short_hash()

    heading = """*******************
Test result summary
*******************

This table contains the results of all tests that were executed in the creation of this test report. Additionally
it contains the git revision that was used as well as the time of execution.

"""

    try:
        passed = report['summary']['passed']
    except KeyError:
        passed = 0
    try:
        failed = report['summary']['failed']
    except KeyError:
        failed = 0
    try:
        skipped = report['summary']['skipped']
    except KeyError:
        skipped = 0
    table = [["Execution date", " ".join([date, time])],
             ["Git revision", commithash],
             [":ref:`Tests passed <Tests passed>`", passed],
             [":ref:`Tests failed <Tests failed>`", failed],
             [":ref:`Tests skipped <Tests skipped>`", skipped],
             ["Tests total", report['summary']['total']]]
    tabulated = tabulate(table, tablefmt='rst')

    footer = """
Click the passed, failed and skipped sections in the table, to skip directly to the corresponding section in the list below.
"""

    return "\n".join([heading, tabulated, footer])


def create_passed_table(passed):
    """Create the table of passed tests, grouped by filename."""

    heading = """
.. _Tests passed:

* Passed tests"""

    body = ""
    for key, value in passed.items():
        value_joined = "\n    * ".join(value)
        body += f"""
  * {key}

    * {value_joined}
"""
    return "\n".join([heading, body])


def create_failed_table(failed):
    """Create the table of failed tests, grouped by filename."""

    heading = """
.. _Tests failed:
    
* Failed tests"""

    body = ""
    for key, value in failed.items():
        value_joined = "\n    * ".join(value)
        body += f"""
      * {key}

        * {value_joined}
    """
    return "\n".join([heading, body])


def create_skipped_table(skipped):
    """Create the table of skipped tests, grouped by filename."""

    heading = """
.. _Tests skipped:

* Skipped tests"""

    body = ""
    for key, value in skipped.items():
        value_joined = "\n    * ".join(value)
        body += f"""
      * {key}

        * {value_joined}
    """
    return "\n".join([heading, body])


def generate():
    print("Parsing test results and generating overview files")
    eradiate_dir = pathlib.Path(os.environ['ERADIATE_DIR'])
    html_report_dir = eradiate_dir / 'test_report'
    build_dir = html_report_dir / "generated"
    if not pathlib.Path.exists(build_dir):
        os.mkdir(build_dir)

    passed = dict()
    failed = dict()
    skipped = dict()

    for file in [f for f in os.listdir(build_dir) if os.path.isfile(os.path.join(build_dir, f))]:
        if os.path.splitext(file)[-1] == ".json":
            with open(os.path.join(build_dir, file), "r") as json_file:
                report = json.load(json_file)
            passed_, failed_, skipped_ = group_results_unittests(report)
            passed = {**passed, **passed_}
            failed = {**failed, **failed_}
            skipped = {**skipped, **skipped_}

    with open(build_dir / "summary.rst", "w") as summary_file:
        summary_file.write(create_summary_table(report))

    with open(build_dir / "passed.rst", "w") as passed_file:
        passed_file.write(create_passed_table(passed))

    with open(build_dir / "failed.rst", "w") as failed_file:
        failed_file.write(create_failed_table(failed))

    with open(build_dir / "skipped.rst", "w") as skipped_file:
        skipped_file.write(create_skipped_table(skipped))


if __name__ == "__main__":
    generate()
