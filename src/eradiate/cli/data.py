import enum
import logging
from typing import Annotated, List, Optional

import typer

from eradiate import asset_manager, fresolver

from ._console import message, section

app = typer.Typer()

logger = logging.getLogger(__name__)


class ListWhat(str, enum.Enum):
    resources = "resources"
    aliases = "aliases"
    all = "all"


class ClearWhat(str, enum.Enum):
    cached = "cached"
    unpacked = "unpacked"
    installed = "installed"
    all = "all"


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Display the asset manager and file resolver configurations.
    Use subcommands for other data management tasks.
    """
    if ctx.invoked_subcommand is None:
        asset_manager_info = asset_manager.info()
        section("Asset manager")
        message(f"• Remote storage URL: {asset_manager_info['remote_url']}")
        message(
            f"• Asset cache location [{asset_manager_info['cache_size']:.3g~P}]: "
            f"{asset_manager_info['cache_dir']}"
        )
        message(
            f"• Unpacked asset location [{asset_manager_info['unpack_size']:.3g~P}]: "
            f"{asset_manager_info['unpack_dir']}"
        )
        message(f"• Installation location: {asset_manager_info['install_dir']}")

        fresolver_info = fresolver.info()
        section("File resolver")
        for path in fresolver_info["paths"]:
            message(f"• {path}")


@app.command()
def update():
    """
    Download the data registry manifest from the remote data location.
    """
    asset_manager.update(download=True)


@app.command()
def list(
    what: Annotated[
        ListWhat,
        typer.Option(help="A keyword that specifies what to clear."),
    ] = ListWhat.resources,
    aliases: Annotated[bool, typer.Option(help="Alias to --what aliases.")] = False,
    all: Annotated[bool, typer.Option(help="Alias to --what all.")] = False,
):
    """
    List all packages referenced by the manifest and their current state
    (cached, unpacked, installed).
    """
    if aliases:
        what = ListWhat.aliases
    if all:
        what = ListWhat.all

    asset_manager.update()
    asset_manager.list(what=what)


@app.command()
def download(
    resource_ids: Annotated[
        List[str], typer.Argument(help="One or multiple resource IDs or aliases.")
    ],
    unpack: Annotated[bool, typer.Option(help="Unpack downloaded archives.")] = True,
):
    """
    Download a resource from remote storage to the cache directory.
    """
    asset_manager.download(resource_ids, unpack=unpack, progressbar=True)


@app.command()
def install(
    resource_ids: Annotated[
        List[str], typer.Argument(help="One or multiple resource IDs or aliases.")
    ],
):
    """
    Install a resource. If the data is not already cached locally, it is
    downloaded from remove storage.
    """
    asset_manager.install(resource_ids)


@app.command()
def remove(
    resource_ids: Annotated[
        List[str], typer.Argument(help="One or multiple resource IDs or aliases.")
    ],
):
    """
    Uninstall a resource.
    """
    asset_manager.remove(resource_ids)


@app.command()
def clear(
    resource_ids: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Resource(s) for which to clear data. If unset, all data will be wiped."
        ),
    ] = None,
    what: Annotated[
        ClearWhat,
        typer.Option(help="A keyword that specifies what to clear."),
    ] = ClearWhat.cached,
    all: Annotated[bool, typer.Option(help="Alias to --what all.")] = False,
    unpacked: Annotated[bool, typer.Option(help="Alias to --what unpacked.")] = False,
    installed: Annotated[bool, typer.Option(help="Alias to --what installed.")] = False,
):
    """
    Delete data.
    """
    if unpacked:
        what = ClearWhat.unpacked
    if installed:
        what = ClearWhat.installed
    if all:
        what = ClearWhat.all

    asset_manager.clear(resource_ids, what=what)
