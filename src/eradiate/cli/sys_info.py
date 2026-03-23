import joseki  # noqa: F401  # Put import at top to mitigate undesired log output
import typer

from eradiate.cli._console import console, make_table

app = typer.Typer()


@app.command()
def main():
    """
    Display information useful for debugging.
    """
    import eradiate.util.sys_info

    from ._console import section, warning

    warnings = eradiate.kernel.check_kernel()

    if warnings:
        section("Warnings", newline=False)
    for w in warnings:
        warning("• " + w)

    sys_info = eradiate.util.sys_info.show()

    def _table():
        return make_table()

    section("System")
    table = _table()
    table.add_row("CPU", sys_info["cpu_info"])
    table.add_row("OS", sys_info["os"])
    table.add_row("Python", sys_info["python"])
    console.print(table)

    section("Versions")
    table = _table()
    table.add_row("eradiate", eradiate.__version__)
    table.add_row("drjit", sys_info["drjit_version"])
    table.add_row(
        "eradiate-mitsuba",
        f"{sys_info['eradiate_mitsuba_version']} "
        f"(based on mitsuba {sys_info['mitsuba_version']})",
    )
    for package in ["numpy", "scipy", "xarray", "axsdb", "joseki"]:
        table.add_row(package, sys_info[package])

    try:
        import mitsuba as mi

        table.add_section()
        table.add_row(
            "Mitsuba variants", "\n".join([variant for variant in mi.variants()])
        )
    except ImportError:
        pass

    console.print(table)

    section("Configuration")
    table = _table()

    for var in ["SOURCE_DIR"]:
        value = getattr(eradiate.config, var)
        var_repr = str(value)
        table.add_row(f"ERADIATE_{var.upper()}", var_repr)

    loaded_settings_files = list(eradiate.config.settings._loaded_files)
    if loaded_settings_files:
        item = "Loaded settings files"
        for fname in loaded_settings_files:
            table.add_row(item, fname)
            item = ""
    else:
        table.add_row("Loaded settings files", "<none>")

    console.print(table)


__doc__ = main.__doc__
