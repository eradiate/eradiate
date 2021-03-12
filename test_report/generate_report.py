import argparse
import os
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

    if isinstance(target_dir, list):
        cmd = cmd + target_dir
    elif target_dir is None:
        pass
    else:
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


def create_pdf(eradiate_dir):
    cmd_sphinx = [
        "python",
        "-m",
        "sphinx",
        "-b",
        "latex",
        os.path.join(eradiate_dir, "test_report"),
        os.path.join(eradiate_dir, "build", "pdf_test-report"),
    ]

    cmd_make = [
        "make",
        "-C",
        os.path.join(eradiate_dir, "build", "pdf_test-report"),
    ]

    execute(cmd_sphinx)
    execute(cmd_make)


def cli():
    helptext = \
        """
        Generate the Eradiate test report by first running all tests available
        in python and collecting their results.

        The --no-pytest option can be used to skip test execution if only
        generation of report documents is desired.
        """

    eradiate_dir = os.environ["ERADIATE_DIR"]
    build_dir = os.path.join(eradiate_dir, "test_report", "generated")
    special_dirs = []  # Now empty, still here in case it's again useful

    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--no-pytest",
        help="Skip pytest execution and only generate HTML report",
        action="store_true",
    )
    parser.add_argument(
        "--pdf",
        help="Create a pdf report instead of html",
        action="store_true"
    )
    args = parser.parse_args()

    if args.no_pytest:
        pass
    else:
        # Currently the Mitsuba2 tests are not part of the test report
        # run_pytest(target_dir=os.path.join(kernel_dir, "src"),
        #            json_report_file=os.path.join(build_dir, 'report_kernel.json'))
        run_pytest(target_dir=os.path.join(eradiate_dir, "eradiate"),
                   json_report_file=os.path.join(build_dir, "report_eradiate.json"))
        run_pytest(target_dir=special_dirs,
                   json_report_file=os.path.join(build_dir, "report_special.json"))

    summaries.generate()
    ts.generate()

    if args.pdf:
        create_pdf(eradiate_dir)
    else:
        create_html(eradiate_dir)


if __name__ == "__main__":
    cli()
