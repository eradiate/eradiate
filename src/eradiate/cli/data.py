import logging
import os.path
import textwrap
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from ruamel.yaml import YAML
from typing_extensions import Annotated

import eradiate

app = typer.Typer()
console = Console(color_system=None)

logger = logging.getLogger(__name__)


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
            "retrieved from the data store or file list specifications. "
            "A file list keyword can also be specified to download a "
            "pre-defined list of files (use the `--list` option to display) "
            "available keywords. "
            "If unset, the list of files is read from a YAML file which can be "
            "specified by using the `--from-file` option, if it is used. "
            "Otherwise, it defaults to `all` in a production environment and "
            "`minimal` in a development environment."
        ),
    ] = None,
    from_file: Annotated[
        Optional[str],
        typer.Option(
            "--from-file",
            "-f",
            help="Optional path to a file list (YAML format). If this option "
            "is set, the FILES argument(s) will be ignored.",
        ),
    ] = None,
    list: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List built-in file lists and exit.",
        ),
    ] = False,
):
    """
    Fetch files from the Eradiate data store.
    """
    from eradiate.data._util import get_file_list, known_file_lists
    from eradiate.exceptions import DataError

    if list:
        print("Known file lists:")
        for file_list in known_file_lists():
            print(f"  {file_list}")
        exit(0)

    if not file_list:
        if from_file is None:
            if eradiate.config.SOURCE_DIR is None:
                file_list = ["all"]
            else:
                file_list = ["minimal"]
        else:
            console.print(f"Reading file list from '{from_file}'")
            yaml = YAML()
            file_list = yaml.load(Path(from_file))

    converted = []
    for spec in file_list:
        try:
            converted.extend(get_file_list(spec))
        except ValueError:
            converted.append(spec)

    file_list = converted

    for filename in file_list:
        try:
            console.print(f"[blue]Fetching '{filename}'[/]")
            path = eradiate.data.data_store.fetch(filename)
        except DataError:
            console.print("[red]✗[/] not found")
        else:
            console.print(f"[green]✓[/] found \[{path}]")


@app.command()
def purge_cache(
    keep: Annotated[
        bool, typer.Option("--keep", "-k", help="Keep registered files.")
    ] = False,
):
    """
    Purge the cache of online data stores.
    """
    for data_store_id, data_store in eradiate.data.data_store.stores.items():
        console.print(
            f"[bold cyan]{data_store_id}[/] [{data_store.__class__.__name__}]"
        )

        if isinstance(data_store, eradiate.data.SafeDirectoryDataStore):
            console.print("  Skipping")
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


@app.command()
def check(
    keywords: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="A keyword defining the datasets that are to be checked. "
            "See the `--list` option for available keywords,"
        ),
    ] = None,
    list: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List known keywords and exit.",
        ),
    ] = False,
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            "-f",
            help="Fix issues that can be.",
        ),
    ] = False,
):
    """
    Check data for availability and integrity, optionally fix them.
    """
    from eradiate.radprops import AbsorptionDatabase
    from eradiate.radprops._absorption import (
        KNOWN_DATABASES as KNOWN_MOLECULAR_ABSORPTION_DATABASES,
    )

    if list:
        print("Known keywords:")
        for key in KNOWN_MOLECULAR_ABSORPTION_DATABASES.keys():
            print(f"  {key}")
        return

    if keywords is None:
        keywords = KNOWN_MOLECULAR_ABSORPTION_DATABASES.keys()

    for key in keywords:
        logger.info(f"Opening '{key}'")
        try:
            AbsorptionDatabase.from_name(key, fix=fix)
            logger.info("Success!")
        except FileNotFoundError:
            pass
