"""
Command-line interface to the :mod:`~eradiate.srf_tools` module.
"""

from pathlib import Path
from typing import Optional

import pint
import typer
import xarray as xr
from rich.console import Console
from typing_extensions import Annotated

app = typer.Typer()
console = Console(color_system=None)


@app.callback()
def main():
    """
    Spectral response function filtering utility.
    """
    pass


@app.command()
def trim(
    filename: Annotated[
        Path,
        typer.Argument(
            # exists=True,
            # help="path to the spectral response function data set to process.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="File where to write the filtered data set."),
    ],
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Display filtering summary."),
    ] = False,
    show_plot: Annotated[
        bool,
        typer.Option("-s", "--show-plot", help="Show plot of the filtered region."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("-d", "--dry-run", help="Do not write filtered data to disk."),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "-i",
            "--interactive",
            help="Prompt user to proceed to saving the filtered dataset.",
        ),
    ] = False,
):
    """
    Trim a spectral response function.
    Remove all-except-last leading zeros and all-except-first trailing zeros.
    """
    from eradiate import srf_tools

    srf_tools.trim_and_save(
        srf=filename,
        path=output,
        verbose=verbose,
        show_plot=show_plot,
        dry_run=dry_run,
        interactive=interactive,
    )


def text_input_to_quantity(
    value: Optional[str], default_units: str = "nm"
) -> Optional[pint.Quantity]:
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
    from eradiate import unit_registry as ureg

    if value is None:
        return None
    else:
        # try to parse value into wavelength quantity
        parsed = ureg(value)
        if isinstance(parsed, pint.Quantity):
            return parsed
        else:  # float
            return ureg.Quantity(parsed, default_units)


@app.command()
def filter(
    filename: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help=" path to the spectral response function data to filter.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            help="Path where to write the filtered data.",
        ),
    ],
    trim: Annotated[
        bool,
        typer.Option(
            help="Trim the data set prior to filtering.",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Display filtering summary.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "-d",
            "--dry-run",
            help="Do not write filtered data to file.",
        ),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "-i",
            "--interactive",
            help="Prompt user to proceed to saving the filtered dataset.",
        ),
    ] = False,
    show_plot: Annotated[
        bool,
        typer.Option(
            "-s",
            "--show-plot",
            help="Show plot of the filtered region.",
        ),
    ] = False,
    threshold: Annotated[
        Optional[float],
        typer.Option(
            "-t",
            "--threshold",
            help="Data points where response is less then or equal to this "
            "value are dropped.",
        ),
    ] = None,
    wmin: Annotated[
        Optional[float],
        typer.Option(
            "--wmin",
            "-w",
            help="Lower wavelength value in nm.",
        ),
    ] = None,
    wmax: Annotated[
        Optional[float],
        typer.Option(
            "--wmax",
            "-W",
            help="Upper wavelength value in nm.",
        ),
    ] = None,
    percentage: Annotated[
        Optional[float],
        typer.Option(
            "-p",
            "--percentage",
            help="Data points that do not contribute to this percentage of the "
            "integrated spectral response are dropped",
        ),
    ] = None,
):
    """
    Filter a spectral response function data set.
    """
    from eradiate import srf_tools

    # input conversion
    input_path = Path(filename).absolute()
    srf = xr.load_dataset(input_path)
    wmin = text_input_to_quantity(value=wmin)
    wmax = text_input_to_quantity(value=wmax)

    # filter
    srf_tools.filter_srf(
        srf=srf,
        path=output,
        trim_prior=trim,
        verbose=verbose,
        show_plot=show_plot,
        dry_run=dry_run,
        interactive=interactive,
        threshold=threshold,
        wmin=wmin,
        wmax=wmax,
        percentage=percentage,
    )
