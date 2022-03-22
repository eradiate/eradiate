"""
Notebook support components.
"""


def install(*extensions):
    """
    Install notebook helpers.

    Parameters
    ----------
    *extensions : str
        Extensions to activate. Available extensions (all are active if this
        parameter is unset):

        * :monobold:`kernel_logging`: Route kernel logs through standard logging
          facilities. Progress display uses `tqdm <https://tqdm.github.io/>`_.
        * :monobold:`rich_pretty`: Install :func:`rich.pretty <rich.pretty.install>`
          with default configuration to the current Python REPL
          (includes iPython and Jupyter sessions).

    Warnings
    --------
    Requires an active mode. You can change modes afterwards.
    """
    from rich.pretty import install as install_rich

    from eradiate.kernel.logging import install_logging

    if not extensions:
        extensions = ("kernel_logging", "rich_pretty")

    if "kernel_logging" in extensions:
        install_logging()

    if "rich_pretty" in extensions:
        install_rich()
