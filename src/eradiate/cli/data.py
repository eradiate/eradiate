import os.path
import textwrap
from pathlib import Path
from typing import List, Optional

import typer
from importlib_resources import files
from rich.console import Console
from ruamel.yaml import YAML
from typing_extensions import Annotated

import eradiate

from ..exceptions import DataError

app = typer.Typer()
console = Console(color_system=None)


@app.callback()
def main():
    """
    Manage data.
    """
    pass


@app.command()
def make_registry(
    input_directory: Annotated[
        str,
        typer.Option(
            "--input-directory",
            "-i",
            help="Path to input directory.",
        ),
    ] = ".",
    output_file: Annotated[
        Optional[str],
        typer.Option(
            "--output-file",
            "-o",
            help="Path to output file (default: '<input_directory>/registry.txt').",
        ),
    ] = None,
    rules: Annotated[
        Optional[str],
        typer.Option(
            "--rules",
            "-r",
            help="Path to the registry rule file "
            "(default: '<input_directory>/registry_rules.yml').",
        ),
    ] = None,
    hash_algorithm: Annotated[
        str,
        typer.Option(
            "--hash-algorithm",
            "-a",
            help="Hashing algorithm (default: 'sha256').",
        ),
    ] = "sha256",
):
    """
    Recursively construct a file registry from the current working directory.
    """
    from eradiate.data._core import load_rules, make_registry

    input_directory = Path(input_directory)
    console.print(f"Creating registry file from '{input_directory}'")

    # Load include and exclude rules
    if rules is None:
        rules = input_directory / "registry_rules.yml"
    console.print(f"Using rules in '{rules}'")
    rule_map = load_rules(rules)

    # Write registry
    if output_file is None:
        output_file = input_directory / "registry.txt"
    console.print(f"Writing registry file to '{output_file}'")
    make_registry(
        input_directory,
        output_file,
        includes=rule_map["include"],
        excludes=rule_map["exclude"],
        alg=hash_algorithm,
        show_progress=True,
    )


@app.command()
def update_registries():
    """
    Update local registries for online sources.
    """
    # Update data store registries
    for data_store_id, data_store in eradiate.data.data_store.stores.items():
        if isinstance(data_store, eradiate.data.SafeOnlineDataStore):
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


@app.command()
def fetch(
    file_list: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="An arbitrary number of relative paths to files to be "
            "retrieved from the data store. If unset, the list of "
            "files is read from a YAML file which can be specified by "
            "using the ``--from-file`` option and defaults to "
            "``$ERADIATE_SOURCE_DIR/data/downloads.yml`` a production "
            "environment and "
            "``$ERADIATE_SOURCE_DIR/data/downloads_development.yml`` in a "
            "development environment."
        ),
    ] = None,
    from_file: Annotated[
        Optional[str],
        typer.Option(
            "--from-file",
            "-f",
            help="Optional path to a file list (YAML format). If this option is set, "
            "the FILES argument(s) will be ignored.",
        ),
    ] = None,
):
    """
    Fetch files from the Eradiate data store.
    """
    if not file_list:
        if from_file is None:
            # TODO: fetch this list from online
            if eradiate.config.source_dir is None:
                from_file = files("eradiate") / "src" / "eradiate" / "data" / "downloads.yml"
            else:
                from_file = (
                    eradiate.config.source_dir / "src" / "eradiate" / "data" / "downloads_development.yml"
                )
        console.print(f"Reading file list from '{from_file}'")
        yaml = YAML()
        file_list = yaml.load(from_file)

    for filename in file_list:
        try:
            console.print(f"[blue]Fetching '{filename}'[/]")
            path = eradiate.data.data_store.fetch(filename)
        except DataError:
            console.print(f"[red]✗[/] not found")
        else:
            console.print(f"[green]✓[/] found \[{path}]")


@app.command()
def purge_cache(
    keep: Annotated[
        bool, typer.Option("--keep", "-k", help="Keep registered files.")
    ] = False
):
    """
    Purge the cache of online data stores.
    """
    for data_store_id, data_store in eradiate.data.data_store.stores.items():
        console.print(
            f"[bold cyan]{data_store_id}[/] [{data_store.__class__.__name__}]"
        )

        if isinstance(data_store, eradiate.data.SafeDirectoryDataStore):
            console.print(f"  Skipping")
            continue

        if isinstance(data_store, eradiate.data.SafeOnlineDataStore):
            console.print(f"  Purging '{data_store.path}'")
            if keep:
                data_store.purge(keep="registered")
            else:
                data_store.purge()
            continue

        if isinstance(data_store, eradiate.data.BlindOnlineDataStore):
            console.print(f"  Purging '{data_store.path}'")
            data_store.purge()
            continue


@app.command()
def info(
    data_stores: Annotated[
        Optional[List[str]],
        typer.Argument(
            help=(
                "List of data stores for which information is requested. "
                "If no data store ID is passed, information is displayed for "
                "all data stores."
            )
        ),
    ] = None,
    list_registry: Annotated[
        bool,
        typer.Option(
            "-l", "--list-registry", help="Show registry content if relevant."
        ),
    ] = False,
):
    """
    Display information about data store configuration.
    """

    # Build section list
    sections = [
        ("[purple]Base URL[/] \[base_url]", "base_url"),
        ("[purple]Local path[/] \[path]", "path"),
        ("[purple]Registry path[/] \[registry_path]", "registry_path"),
    ]

    if list_registry:
        sections.append(("Registered files", "registry_keys"))

    first = True

    for data_store_id, data_store in eradiate.data.data_store.stores.items():
        if data_stores and (data_store_id not in data_stores):
            continue

        # Collect section contents
        reprs = {}

        for attr in ["base_url", "path", "registry_path"]:
            try:
                reprs[attr] = f"'{getattr(data_store, attr)}'"
            except (NotImplementedError, AttributeError):
                reprs[attr] = None

        try:
            reprs["registry_keys"] = "\n".join(
                f"'{x}'" for x in sorted(data_store.registry.keys())
            )
        except (NotImplementedError, AttributeError):
            reprs["registry_keys"] = None

        # Display the content for current data store
        if not first:
            console.print()
        else:
            first = False

        console.print(
            f"[bold cyan]{data_store_id}[/] \[{data_store.__class__.__name__}]"
        )

        for section_title, repr_key in sections:
            the_repr = reprs[repr_key]

            if the_repr is not None:
                console.print(f"  {section_title}")
                console.print(textwrap.indent(reprs[repr_key], " " * 4))
