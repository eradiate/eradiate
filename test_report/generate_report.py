import argparse
import os
from shutil import copyfile
import subprocess

import generate_summary_files as summaries
import parse_test_spec as ts


def execute(cmd, **kwargs):
    p = subprocess.Popen(cmd, **kwargs)
    p.wait()


def run_pytest(target_dir=None, json_report_file=None, cwd=None):
    """Run pytest in target_dir (if specified) and generate a JSON test report
    (if json_report_file is passed). The optional cwd argument can be use to 
    set subprocess.Popen()'s cmd keyword argument.
    """
    cmd = ["python", "-m", "pytest"]

    if target_dir is not None:
        cmd.append(target_dir)

    if json_report_file is not None:
        cmd.extend(["--json-report", f"--json-report-file={json_report_file}"])

    kwargs = {}
    if cwd is not None:
        kwargs["cwd"] = cwd

    execute(cmd, **kwargs)


def create_html(eradiate_dir):
    cmd = [
        "python",
        "-m",
        "sphinx",
        "-b",
        "html",
        os.path.join(eradiate_dir, "test_report"),
        os.path.join(eradiate_dir, "build", "html_test-report"),
    ]

    execute(cmd)


def cli():
    helptext = \
        """
        Generate the Eradiate test report by first running all tests available
        in python and collecting their results.

        The --no-pytest option can be used to skip test execution if only
        generation of report documents is desired.
        """

    eradiate_dir = os.environ["ERADIATE_DIR"]
    kernel_dir = os.environ["MITSUBA_DIR"]
    build_dir = os.path.join(eradiate_dir, "test_report", "generated")

    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--no-pytest",
        help="Skip pytest execution and only generate HTML report",
        action="store_true",
    )
    args = parser.parse_args()

    if args.no_pytest:
        pass
    else:
        run_pytest(target_dir=os.path.join(kernel_dir, "src"),
                   json_report_file=os.path.join(build_dir, 'report_kernel.json'))
        run_pytest(target_dir=os.path.join(eradiate_dir, "eradiate"),
                   json_report_file=os.path.join(build_dir, 'report_eradiate.json'))

    summaries.generate()
    ts.generate()

    create_html(eradiate_dir)


if __name__ == "__main__":
    cli()
