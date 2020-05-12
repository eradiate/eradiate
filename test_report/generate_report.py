import argparse
import os
from shutil import copyfile
import subprocess

import generate_summary_files as summaries
import parse_test_spec as ts


def run_kernel_tests(kernel_dir, build_dir):
    """
    Run the tests or eradiate.kernel, by calling the cmake target.
    Afterwards, copy the report to test_report/generated for further processing
    """
    p = subprocess.Popen(
        [
            "python",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={os.path.join(build_dir, 'report_kernel.json')}",
        ],
        cwd=kernel_dir,
    )
    p.wait()


def run_eradiate_tests(eradiate_dir, build_dir):
    """
    Run the pytest test-suite for the eradiate tests, creating a report at
    test_report/generated
    """
    p = subprocess.Popen(
        [
            "python",
            "-m",
            "pytest",
            "eradiate",
            "--json-report",
            f"--json-report-file={os.path.join(build_dir, 'report_eradiate.json')}",
        ],
        cwd=eradiate_dir,
    )
    p.wait()


def create_html(eradiate_dir):
    p = subprocess.Popen(
        [
            "python",
            "-m",
            "sphinx",
            "-b",
            "html",
            os.path.join(eradiate_dir, "test_report"),
            os.path.join(eradiate_dir, "build", "html_test-report"),
        ]
    )
    p.wait()


if __name__ == "__main__":
    docstring = """
    Generate the Eradiate test report by first running all tests available
    in python and collecting their results.

    The `--no-execute` option can be used to skip test execution if only
    generation of report documents is desired.
    """
    
    eradiate_dir = os.environ["ERADIATE_DIR"]
    kernel_dir = os.environ["MITSUBA_DIR"]
    build_dir = os.path.join(eradiate_dir, "test_report", "generated")

    parser = argparse.ArgumentParser(description=docstring)
    parser.add_argument(
        "--no_execute",
        help="Do not execute tests before building the report",
        action="store_true",
    )
    args = parser.parse_args()

    if args.no_execute:
        pass
    else:
        run_kernel_tests(kernel_dir, build_dir)
        run_eradiate_tests(eradiate_dir, build_dir)

    summaries.generate()
    ts.generate()

    create_html(eradiate_dir)
