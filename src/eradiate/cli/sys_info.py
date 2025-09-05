import joseki  # noqa: F401  # Put import at top to mitigate undesired log output
import typer

app = typer.Typer()


@app.command()
def main():
    """
    Display information useful for debugging.
    """
    import eradiate.util.sys_info

    from ._console import message, section, warning

    warnings = eradiate.kernel.check_kernel()

    if warnings:
        section("Warnings", newline=False)
    for w in warnings:
        warning("• " + w)

    sys_info = eradiate.util.sys_info.show()
    section("System")
    message(f"CPU: {sys_info['cpu_info']}")
    message(f"OS: {sys_info['os']}")
    message(f"Python: {sys_info['python']}")

    section("Versions")
    message(f"• eradiate {eradiate.__version__}")
    message(f"• drjit {sys_info['drjit_version']}")
    message(f"• mitsuba {sys_info['mitsuba_version']}")
    message(f"• eradiate-mitsuba {sys_info['eradiate_mitsuba_version']}")
    message(f"• numpy {sys_info['numpy']}")
    message(f"• scipy {sys_info['scipy']}")
    message(f"• xarray {sys_info['xarray']}")

    try:
        import mitsuba as mi

        section("Available Mitsuba variants")
        message("\n".join([f"• {variant}" for variant in mi.variants()]))
    except ImportError:
        pass

    section("Configuration")
    for var in ["SOURCE_DIR"]:
        value = getattr(eradiate.config, var)
        var_repr = str(value)
        message(f"• ERADIATE_{var.upper()}: {var_repr}")

    loaded_settings_files = list(eradiate.config.settings._loaded_files)
    if loaded_settings_files:
        message("• Loaded settings files:")
        for fname in loaded_settings_files:
            message(f"  • {fname}")
    else:
        message("• Loaded setting files: <none>")


__doc__ = main.__doc__
