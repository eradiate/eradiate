import mitsuba as mi
import typer
from rich.console import Console

app = typer.Typer()




@app.command()
def main():
    """
    Display information useful for debugging.
    """
    import eradiate.util.sys_info

    console = Console(color_system=None)
    error_console = Console(stderr=True, color_system=None)
    sys_info = eradiate.util.sys_info.show()

    def section(title, newline=True):
        if newline:
            console.print()
        console.rule("── " + title, align="left")
        console.print()

    def message(text):
        console.print(text)

    def warning(text):
        error_console.print(text)

    from eradiate.kernel import (
        ERADIATE_KERNEL_VERSION,
        _EXPECTED_MITSUBA_VERSION,
        ERADIATE_KERNEL_PATCH_VERSION,
        _EXPECTED_MITSUBA_PATCH_VERSION
    )

    warnings = []
    # Check if the kernel version is compatible
    if ERADIATE_KERNEL_VERSION != _EXPECTED_MITSUBA_VERSION:
        warnings.append(
            "Using an incompatible version of Mitsuba. Eradiate requires Mitsuba "
            f"{_EXPECTED_MITSUBA_VERSION}. Found Mitsuba {ERADIATE_KERNEL_VERSION}."
        )
    if ERADIATE_KERNEL_PATCH_VERSION is None:
        warnings.append(
            "Using a without explicit support for Eradiate. Make sure your kernel "
            "is compiled with the appropriate variants and plugins to run Eradiate."
        )
    elif ERADIATE_KERNEL_PATCH_VERSION != _EXPECTED_MITSUBA_PATCH_VERSION:
        warnings.append(
            "Using an incompatible patch version of Mitsuba. Eradiate requires a "
            f"Mitsuba kernel version {_EXPECTED_MITSUBA_VERSION}, with a specific patch "
            f"version {_EXPECTED_MITSUBA_PATCH_VERSION}. Found Mitsuba "
            f"{ERADIATE_KERNEL_VERSION} with the patch version"
            f" {ERADIATE_KERNEL_PATCH_VERSION}."
        )

    if warnings:
        section("Warnings", newline=False)
    for w in warnings:
        warning("• " + w)

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

    section("Available Mitsuba variants")
    message("\n".join([f"• {variant}" for variant in mi.variants()]))

    section("Configuration")
    for var in sorted(x.name for x in eradiate.config.__attrs_attrs__):
        value = getattr(eradiate.config, var)
        if var == "progress":
            var_repr = f"{str(value)} ({value.value})"
        else:
            var_repr = str(value)
        message(f"• ERADIATE_{var.upper()}: {var_repr}")


__doc__ = main.__doc__
