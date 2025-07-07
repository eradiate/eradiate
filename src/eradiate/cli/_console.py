from rich.console import Console

console = Console(color_system=None)
error_console = Console(stderr=True, color_system=None)


def section(title, newline=True):
    if newline:
        console.print()
    console.rule("── " + title, align="left")
    console.print()


def message(text):
    console.print(text)


def warning(text):
    error_console.print(text)
