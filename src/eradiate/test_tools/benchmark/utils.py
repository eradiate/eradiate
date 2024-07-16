import subprocess


def get_command_output(cmd):
    """
    Run a command in the shell and return the output.
    Will raise and display any errors too.

    Parameters
    ----------
    cmd : str
        Command to run in shell, formatted as used in shell directly.

    Returns
    -------
    str
        Stdout of the process run in shell.
    """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout = result.stdout.decode().strip()
    if result.returncode == 0:
        return stdout

    stderr = result.stderr.decode().strip()
    raise RuntimeError(
        "Problem running '%s' (STDOUT: '%s' STDERR: '%s')" % (cmd, stdout, stderr)
    )


def get_repo_info():
    """
    Returns
    -------
    str
        The remote repo url.
    str
        The branch currently checked out.
    """
    out = get_command_output("git remote -v")
    repo = out.split("\n")[-1].split()[1]
    branch = get_command_output("git rev-parse --abbrev-ref HEAD")
    return (repo, branch)


def get_commit_info():
    """
    Get currently checked out git commit hash and time of commit.

    Returns
    -------
    str
        Commit hash currently checked out.
    str
        Time of commit.
    """
    commitHash = get_command_output("git rev-parse HEAD")
    commitTime = get_command_output("git log -n1 --pretty=%%ct %s" % commitHash)
    return (commitHash, str(int(commitTime) * 1000))
