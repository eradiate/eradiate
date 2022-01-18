import click
from rich.console import Console

import eradiate

console = Console()


@click.command()
def cli():
    """
    Eradiate - Display information useful for debugging.
    """
    console.rule("Version")
    console.print(f"• eradiate {eradiate.__version__}")

    console.rule("Configuration")
    for var in [x.name for x in eradiate.config.__attrs_attrs__]:
        console.print(f"• ERADIATE_{var.upper()}: {getattr(eradiate.config, var)}")

    console.rule("Path resolver")
    for path in eradiate.path_resolver:
        console.print(f"• {path}")


if __name__ == "__main__":
    cli()
