import logging
import os
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from ruamel.yaml import YAML

import eradiate
import eradiate.data
from eradiate.exceptions import DataError

logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]),
    default="WARNING",
    help="Set log level (default: 'WARNING').",
)
def cli(log_level):
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.option(
    "--input-directory",
    "-i",
    default=".",
    help="Path to input directory (default: '.').",
)
@click.option(
    "--output-file",
    "-o",
    default=None,
    help="Path to output file (default: '<input_directory>/registry.txt').",
)
@click.option(
    "--rules",
    "-r",
    default=None,
    help="Path to the registry rule file (default: '<input_directory>/registry_rules.yml').",
)
@click.option(
    "--hash-algorithm",
    "-a",
    default="sha256",
    help="Hashing algorithm (default: 'sha256').",
)
def make_registry(input_directory, output_file, rules, hash_algorithm):
    """
    Recursively construct a file registry from a directory.
    """
    input_directory = Path(input_directory)
    console.print(f"Creating registry file from '{input_directory}'")

    # Load include and exclude rules
    if rules is None:
        rules = input_directory / "registry_rules.yml"
    console.print(f"Using rules in '{rules}'")
    rule_map = eradiate.data._core.load_rules(rules)

    # Write registry
    if output_file is None:
        output_file = input_directory / "registry.txt"
    console.print(f"Writing registry file to '{output_file}'")
    eradiate.data._core.make_registry(
        input_directory,
        output_file,
        includes=rule_map["include"],
        excludes=rule_map["exclude"],
        alg=hash_algorithm,
        show_progress=True,
    )


@cli.command()
def update_registries():
    """
    Update local registries for online sources.
    """
    # Update data store registries
    for data_store_id, data_store in eradiate.data.data_store.stores.items():
        if isinstance(data_store, eradiate.data.OnlineDataStore):
            console.print(
                f"[bold cyan]{data_store_id}[/] [{data_store.__class__.__name__}]"
            )
            console.print(f"    Refreshing registry \[{data_store.registry_path}]")
            data_store.registry_delete()
            console.print(
                "    Downloading from "
                f"\[{os.path.join(data_store.base_url, data_store.registry_fname)}]"
            )
            data_store.registry_fetch()
            console.print("  [green]✓ Done[/]")
            continue


@cli.command()
@click.argument("files", nargs=-1)
@click.option(
    "--from-file",
    "-f",
    default=None,
    help="Optional path to a file list (YAML format).",
)
def fetch(files, from_file):
    """
    Fetch files from the Eradiate data stores.
    """
    if not files:
        if from_file is None:
            # TODO: fetch this list from online
            from_file = eradiate.config.dir / "resources/downloads.yml"
        console.print(f"Reading file list from '{from_file}'")
        yaml = YAML()
        files = yaml.load(from_file)

    for filename in files:
        try:
            console.print(f"[blue]Fetching '{filename}'[/]")
            path = eradiate.data._store.data_store.fetch(filename)
        except DataError:
            console.print(f"[red]✗[/] not found")
        else:
            console.print(f"[green]✓[/] found \[{path}]")


@cli.command()
@click.option("--keep", "-k", is_flag=True, help="Keep registered files.")
def purge_cache(keep):
    """
    Purge the cache of online data stores.
    """
    for i, data_store in enumerate(eradiate.data._store.data_store.stores, start=1):
        if isinstance(data_store, eradiate.data.DirectoryDataStore):
            continue

        if isinstance(data_store, eradiate.data.OnlineDataStore):
            console.print(f"Purging '{data_store.path}'")
            if keep:
                data_store.purge(keep="registered")
            else:
                data_store.purge()
            continue

        if isinstance(data_store, eradiate.data.BlindDataStore):
            console.print(f"Purging '{data_store.path}'")
            data_store.purge()
            continue


@cli.command()
def info():
    """
    Display information about data store configuration.
    """
    # Print data store information to terminal
    for data_store_id, data_store in eradiate.data._store.data_store.stores.items():
        console.print(
            f"[bold cyan]{data_store_id}[/] [{data_store.__class__.__name__}]"
        )

        if isinstance(data_store, eradiate.data.DirectoryDataStore):
            console.print(f"    path='{data_store.path}'")
            console.print(f"    registry_path='{data_store.registry_path}'")
            continue

        if isinstance(data_store, eradiate.data.OnlineDataStore):
            console.print(f"    base_url='{data_store.base_url}'")
            console.print(f"    path='{data_store.path}'")
            console.print(f"    registry_path='{data_store.registry_path}'")
            continue

        if isinstance(data_store, eradiate.data.BlindDataStore):
            console.print(f"    base_url='{data_store.base_url}'")
            console.print(f"    path='{data_store.path}'")
            continue


if __name__ == "__main__":
    cli()
