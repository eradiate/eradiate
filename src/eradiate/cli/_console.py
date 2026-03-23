from rich.box import Box
from rich.console import Console
from rich.table import Table

console = Console(color_system=None)
error_console = Console(stderr=True, color_system=None)

BOX_COLON = Box(
    "    \n"  # top
    "  : \n"  # header row
    "    \n"  # head/body separator
    "  : \n"  # body row
    "    \n"  # row separator
    "    \n"  # foot separator
    "  : \n"  # footer row
    "    \n"  # bottom
)


def make_table(**kwargs) -> Table:
    """Create a two-column key/value table with colon separator styling."""
    t = Table(
        show_header=False,
        show_lines=False,
        show_edge=False,
        title_justify="left",
        box=BOX_COLON,
        **kwargs,
    )
    t.add_column("Item", justify="right", no_wrap=True)
    t.add_column("Value", justify="left", overflow="fold")
    return t


def section(title, newline=True):
    if newline:
        console.print()
    console.print(title)
    console.print("=" * len(title))
    console.print()


def message(text):
    console.print(text)


def warning(text):
    error_console.print(text)
