"""
The Eradiate command-line interface, built with Typer and Rich.
"""

import logging
from enum import Enum

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from . import data, srf, sys_info


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


app = typer.Typer(
    help="Eradiate — A modern radiative transfer model for Earth observation.",
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    log_level: Annotated[
        LogLevel, typer.Option(help="Set log level.")
    ] = LogLevel.WARNING,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode. This will notably print exceptions with locals.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Display version information and exit.",
        ),
    ] = False,
):
    if version:
        from eradiate import __version__

        print(f"eradiate version {__version__}")
        exit(0)

    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        ctx.exit()

    if debug:
        app.pretty_exceptions_enable = True

    logging.basicConfig(
        level=log_level.name,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


app.command(name="sys-info", help=sys_info.__doc__)(sys_info.main)
app.command(name="show", help="Alias to 'sys-info' (deprecated).")(sys_info.main)
app.add_typer(data.app, name="data")
app.add_typer(srf.app, name="srf")


def main():
    app()


if __name__ == "__main__":
    app()
