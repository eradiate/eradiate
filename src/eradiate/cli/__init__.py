"""
The Eradiate command-line interface, built with Click and Rich.
"""

import logging

import click
from rich.logging import RichHandler

from .data import data
from .show import show


@click.group()
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]),
    default="WARNING",
    help="Set log level (default: 'WARNING').",
)
def main(log_level):
    """
    Eradiate — A modern radiative transfer model for Earth observation.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


main.add_command(show)
main.add_command(data)


if __name__ == "__main__":
    main()
