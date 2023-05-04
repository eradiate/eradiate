"""
The Eradiate command-line interface, built with Click and Rich.
"""

import logging
from enum import Enum

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from . import data, show, srf


class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"


app = typer.Typer(
    help="Eradiate — A modern radiative transfer model for Earth observation."
)


@app.callback()
def cli(
    log_level: Annotated[
        LogLevel, typer.Option(help="Set log level.")
    ] = LogLevel.WARNING
):
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


app.command(name="show", help=show.__doc__)(show.main)
app.add_typer(data.app, name="data")
app.add_typer(srf.app, name="srf")


def main():
    app()


if __name__ == "__main__":
    app()
