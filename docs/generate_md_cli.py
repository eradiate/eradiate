from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from click import Command, Group

import eradiate.cli


def get_docs_for_click_markdown(
    *,
    obj: Command,
    ctx: typer.Context,
    indent: int = 0,
    name: str = "",
    call_prefix: str = "",
) -> str:
    docs = "#" * (1 + indent)
    command_name = name or obj.name
    if call_prefix:
        command_name = f"{call_prefix} {command_name}"
    title = f"`{command_name}`" if command_name else "CLI"
    docs += f" {title}\n\n"
    if obj.help:
        docs += f"{obj.help}\n\n"
    usage_pieces = obj.collect_usage_pieces(ctx)
    if usage_pieces:
        docs += "**Usage**:\n\n"
        docs += "```console\n"
        docs += "$ "
        if command_name:
            docs += f"{command_name} "
        docs += f"{' '.join(usage_pieces)}\n"
        docs += "```\n\n"
    args = []
    opts = []
    for param in obj.get_params(ctx):
        rv = param.get_help_record(ctx)
        if rv is not None:
            if param.param_type_name == "argument":
                args.append(rv)
            elif param.param_type_name == "option":
                opts.append(rv)
    if args:
        docs += f"**Arguments**:\n\n"
        for arg_name, arg_help in args:
            docs += f"* `{arg_name}`"
            if arg_help:
                docs += f": {arg_help}"
            docs += "\n"
        docs += "\n"
    if opts:
        docs += f"**Options**:\n\n"
        for opt_name, opt_help in opts:
            docs += f"* `{opt_name}`"
            if opt_help:
                docs += f": {opt_help}"
            docs += "\n"
        docs += "\n"
    if obj.epilog:
        docs += f"{obj.epilog}\n\n"
    if isinstance(obj, Group):
        group: Group = cast(Group, obj)
        commands = group.list_commands(ctx)
        if commands:
            docs += f"**Commands**:\n\n"
            for command in commands:
                command_obj = group.get_command(ctx, command)
                assert command_obj
                docs += f"* `{command_obj.name}`"
                command_help = command_obj.get_short_help_str()
                if command_help:
                    docs += f": {command_help}"
                docs += "\n"
            docs += "\n"
        for command in commands:
            command_obj = group.get_command(ctx, command)
            assert command_obj
            use_prefix = ""
            if command_name:
                use_prefix += f"{command_name}"
            docs += get_docs_for_click_markdown(
                obj=command_obj, ctx=ctx, indent=indent + 1, call_prefix=use_prefix
            )
    return docs


def docs(typer_obj: typer.Typer, name: str = "", indent: int = 0) -> str:
    """
    Generate Markdown docs for a Typer app.
    Code adapted from the typer-cli project.
    """
    click_obj = typer.main.get_command(typer_obj)
    ctx = typer.Context(click_obj)
    docs = get_docs_for_click_markdown(obj=click_obj, ctx=ctx, name=name, indent=indent)
    clean_docs = f"{docs.strip()}\n"
    return clean_docs


def write_if_modified(filename, content):
    filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filename, "r") as f:
            existing = f.read()
    except OSError:
        existing = None

    if existing == content:
        print(f"Skipping unchanged '{filename.name}'")

    else:
        print(f"Generating '{filename.name}'")
        with open(filename, "w") as f:
            f.write(content)


def generate():
    root_dir = Path(__file__).absolute().parent.parent
    out_dir = root_dir / "docs/src"
    print(f"Generating CLI docs in '{out_dir}'")

    docs_markdown = (
        "(sec-reference_cli)=\n"
        "# Command-line interface reference\n\n"
        "This is the reference for Eradiate’s command-line tools. It consists "
        "of a main entry point `eradiate`, and its multiple subcommands "
        "documented hereafter. The implementation is located in the "
        "`eradiate.cli` module (not documented here).\n\n"
        + docs(eradiate.cli.app, name="eradiate", indent=1)
    )
    write_if_modified(out_dir / "reference_cli.md", docs_markdown)


if __name__ == "__main__":
    generate()
