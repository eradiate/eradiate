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

        * :monobold:`rich_pretty`: Install :func:`rich.pretty <rich.pretty.install>`
          with default configuration to the current Python REPL.
        * :monobold:`rich_traceback`:
          Install :func:`rich.traceback <rich.traceback.install>`
          with default configuration to the current Python REPL.
    """
    from rich.pretty import install as install_rich_pretty
    from rich.traceback import install as install_rich_traceback

    if not extensions:
        extensions = {"rich_pretty", "rich_traceback"}

    if "rich_pretty" in extensions:
        install_rich_pretty()

    if "rich_traceback" in extensions:
        install_rich_traceback()


# IPython extension. Must be imported by top-level module.
def load_ipython_extension(ipython):
    """
    The Eradiate notebook extension.

    This extension simply calls the :func:`eradiate.notebook.install` function.

    See Also
    --------
    :func:`eradiate.notebook.install`

    Notes
    -----
    This extension should be loaded using the IPython magic:

    .. code::

       %load_ext eradiate
    """

    install()
