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
    sys_info = eradiate.util.sys_info.show()

    def section(title, newline=True):
        if newline:
            console.print()
        console.rule("── " + title, align="left")
        console.print()

    def message(text):
        console.print(text)

    section("System", newline=False)
    message(f"CPU: {sys_info['cpu_info']}")
    message(f"OS: {sys_info['os']}")
    message(f"Python: {sys_info['python']}")

    section("Versions")
    message(f"• eradiate {eradiate.__version__}")
    message(f"• drjit {sys_info['drjit_version']}")
    message(f"• mitsuba {sys_info['mitsuba_version']}")
    message(f"• eradiate-mitsuba {sys_info['eradiate_mitsuba_version']}")

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
