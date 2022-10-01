"""
Command-line interface to the :mod:`~eradiate.srf_tools` module.
"""

import typing as t
from pathlib import Path

import click
import pint
import xarray as xr
from rich.console import Console

from eradiate import srf_tools
from eradiate.units import unit_registry as ureg

console = Console(color_system=None)


@click.group()
def srf():
    """
    Spectral response function filtering utility.
    """
    pass


@srf.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-v", "--verbose", is_flag=True, help="Display filtering summary.")
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    help="Prompt before writing filtered data to disk.",
)
@click.option(
    "-d", "--dry-run", is_flag=True, help="Do not write filtered data to disk."
)
def trim(filename, output, verbose, interactive, dry_run):
    """
    Trim a spectral response function.

    Remove all-except-last leading zeros and all-except-first trailing zeros.

    FILENAME is the path to the spectral response function data set to process.
    OUTPUT specifies where to write the filtered data set.
    """
    srf_tools.trim_and_save(
        srf=filename,
        path=output,
        verbose=verbose,
        interactive=interactive,
        dry_run=dry_run,
    )


def text_input_to_quantity(
    value: t.Union[None, str], default_units: str = "nm"
) -> t.Optional[pint.Quantity]:
    """
    Converts text input to wavelength quantity.

    Parameters
    ----------
    value: str or None
        Value to convert to quantity.

    default_units: str, optional
        If ``value`` converts to a dimensionless quantity, use these units.

    Returns
    -------
    quantity, optional
        Converted quantity.
    """
    if value is None:
        return None
    else:
        # try to parse value into wavelength quantity
        parsed = ureg(value)
        if isinstance(parsed, pint.Quantity):
            return parsed
        else:  # float
            return ureg.Quantity(parsed, default_units)


@srf.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--trim/--no-trim",
    default=True,
    show_default=True,
    help="Trim the data set prior to filtering.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Display filtering summary.",
)
@click.option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="Do not write filtered data to file.",
)
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    help="Prompt before writing filtered data to disk.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    help="Data points where response is less then or equal to this value are dropped.",
)
@click.option(
    "-w",
    "--wmin",
    help="Lower wavelength value [float, str]",
)
@click.option(
    "-W",
    "--wmax",
    help="Upper wavelength value [float, str]",
)
@click.option(
    "-p",
    "--percentage",
    type=float,
    help="Data points that do not contribute to this percentage of the integrated spectral response are dropped",
)
def filter(
    filename,
    output,
    trim,
    verbose,
    dry_run,
    interactive,
    threshold,
    wmin,
    wmax,
    percentage,
):
    """
    Filter a spectral response function data set.

    FILENAME specifies the path to the spectral response function data to filter.
    OUTPUT specified the path where to write the filtered data.
    """
    # input conversion
    input_path = Path(filename).absolute()
    srf = xr.load_dataset(input_path)
    wmin = text_input_to_quantity(value=wmin)
    wmax = text_input_to_quantity(value=wmax)

    # filter
    srf_tools.filter(
        srf=srf,
        path=output,
        trim_prior=trim,
        verbose=verbose,
        interactive=interactive,
        dry_run=dry_run,
        threshold=threshold,
        wmin=wmin,
        wmax=wmax,
        percentage=percentage,
    )
