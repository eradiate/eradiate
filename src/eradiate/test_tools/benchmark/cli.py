import os
import platform
import subprocess

import click
import json5

import eradiate

from . import utils
from .asvdb import ASVDb


@click.command
@click.option(
    "--set-commit-hash",
    default=None,
    help="Set the commit hash to use when recording benchmark results. "
    "By default, will use the currently checked out environment for dev environments "
    "and the eradiate version tag's commit for prod environment. Note also "
    "that if git-ref is specified, this will overwrite it when recording the results.",
)
@click.option(
    "--archive-dir",
    default=None,
    help="Specify the directory to another ASV database in which the result should be archived. "
    "The path should point to the root of the database (where the asv.conf.json lives).",
)
# NM - 25/09/24: Disable ASV git cloning features broken by the migration to pixi.
# @click.option(
#     "--git-ref",
#     default=None,
#     help="Specifies a range at which benchmarks should be run. Note that this will run using ASV's "
#     "contained environments and clone and build the project at each commit in the range.",
# )
@click.option("-v", "--verbose", is_flag=True, help="increase verbosity")
@click.option(
    "-e",
    "--show-stderr",
    is_flag=True,
    help="Display the stderr output from the benchmarks.",
)
@click.option(
    "-q",
    "--quick",
    is_flag=True,
    help="Do a 'quick' run, where each benchmark function is run only once. "
    "This is useful to find basic errors in the benchmark "
    "functions faster. The results are unlikely to be useful, "
    "and thus are not saved.",
)
@click.option(
    "-m",
    "--machine",
    default=None,
    help="Use the given name to retrieve machine information. If not provided, "
    "the hostname is used. If no entry with that name is found, and there "
    "is only one entry in ~/.asv-machine.json, that one entry will be used.",
)
def benchmark(
    set_commit_hash, archive_dir, git_ref, verbose, show_stderr, quick, machine
):
    """
    Benchmark test suite based on ASV. By default, runs the test suite on the
    current environment. Specify a git-ref range to run the benchmark on
    cloned projects (ASV's default).

    Warning! Production environment cannot be tested yet because no
    packaged versions contain those changes yet.
    """

    # get current benchmark directory
    eradiate_dir = os.environ.get("ERADIATE_SOURCE_DIR")
    benchmark_dir = os.path.join(eradiate_dir, "benchmarks")
    conf_path = os.path.join(benchmark_dir, "asv.conf.json")
    branches = []
    conf = None
    with open(conf_path) as conf_file:
        conf = json5.load(conf_file)
        branches = conf["branches"]

    cmd = ["asv", "run"]

    # NM - 25/09/24: Disable ASV git cloning features broken by the migration to pixi.
    # if git_ref:
    #     cmd.append(git_ref)
    # else:

    cmd.append("-E")
    cmd.append("existing:same")

    # Pass arguments to ASV
    if verbose:
        cmd.append("-v")

    if show_stderr:
        cmd.append("-e")

    if quick:
        cmd.append("-q")

    if machine:
        cmd.append(f"-m={machine}")

    # By default, dev environment will use the checked out commit hash
    # and prod the commit corresponding to their version tag.
    # if set_commit_hash is None and git_ref is None:
    if set_commit_hash is None:
        version = eradiate.__version__
        if "dev" in version:
            set_commit_hash, _ = utils.get_commit_info()
        else:
            cmd = f"git rev-list -n 1 v{version}"
            if verbose:
                click.echo(f"Running {cmd}")
            set_commit_hash = utils.get_command_output(cmd)

    if set_commit_hash:
        cmd.append(f"--set-commit-hash={set_commit_hash}")

        commit_branches_raw = utils.get_command_output(
            f"git branch --contains {set_commit_hash} "
        )

        commit_branches_raw = commit_branches_raw.replace("*", "")
        commit_branches_raw = commit_branches_raw.replace(" ", "")
        commit_branches = commit_branches_raw.split("\n")

        branch_in_conf = False
        for branch in branches:
            if branch in commit_branches:
                branch_in_conf = True

        if not branch_in_conf:
            if click.confirm(
                f"The branch '{commit_branches[0]}' you are about to benchmark isn't tracked by asv.conf.json. Are you sure you want to continue?",
                abort=True,
            ):
                pass

    # construct and run the full command
    cmd = " ".join(cmd)
    if verbose:
        click.echo(f"Running : {cmd}")
    subprocess.run(cmd, shell=True)

    if archive_dir and not quick:
        if verbose:
            click.echo("Archiving results:")

        if machine is None:
            # get default machine name if none were given
            uname = platform.uname()
            machine = uname.node

        if verbose:
            click.echo(f"\t Loading current database at {benchmark_dir}")

        # get all results in database
        current_db = ASVDb(benchmark_dir)
        results = current_db.getResults()

        if verbose:
            click.echo(f"\t Loading target database at {archive_dir}")

        (repo, branch) = utils.get_repo_info()
        target_db = ASVDb(archive_dir, repo=repo, branches=[branch])

        commit_list = []

        if set_commit_hash:
            commit_list.append(set_commit_hash[:8])

        # NM - 25/09/24: Disable ASV git cloning features broken by the migration to pixi.
        # elif git_ref:
        #     commit_list_raw = utils.get_command_output(f"git rev-list {git_ref}")
        #     commit_list_raw = commit_list_raw.split("\n")
        #     commit_list = [commit[:8] for commit in commit_list_raw]

        # filter by machine name and commit hash
        for result in results:
            info = result[0]
            if info.machineName == machine and info.commitHash[:8] in commit_list:
                if verbose:
                    click.echo(
                        f"\t Archiving {info.commitHash[:8]}-{info.envName}.json"
                    )
                target_db.addResults(info, result[1])


if __name__ == "__main__":
    benchmark()
