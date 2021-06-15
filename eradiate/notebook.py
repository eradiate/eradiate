import rich.pretty

from . import kernel


def install(*extensions):
    """
    Install notebook helpers.

    .. warning:: Requires an active mode. You can change modes afterwards.

    Parameter ``extensions`` (list[str]):
        List of extensions to activate. Available extensions (all are active
        if none is passed):

        * :monobold:`kernel_logging`: Route kernel logs through standard logging
          facilities. Progress display uses `tqdm <https://tqdm.github.io/>`_.
        * :monobold:`rich_pretty`: Install :func:`rich.pretty <rich.pretty.install>`
          with default configuration to the current Python REPL
          (includes iPython and Jupyter sessions).
    """
    if not extensions:
        extensions = ("kernel_logging", "rich_pretty")

    if "kernel_logging" in extensions:
        kernel.logging.install_logging()

    if "rich_pretty" in extensions:
        rich.pretty.install()
