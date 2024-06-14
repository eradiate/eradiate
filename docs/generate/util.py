from pathlib import Path

import jinja2

DOCS_ROOT_DIR = (Path(__file__) / "../..").resolve()
LOCAL_DIR = Path(__file__).parent


def savefig(fig, filename: Path, **kwargs):
    filename.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filename, **kwargs)


def write_if_modified(filename, content):
    filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filename, "r") as f:
            existing = f.read()
    except OSError:
        existing = None

    if existing == content:
        print(f"Skipping unchanged '{filename}'")

    else:
        print(f"Writing to '{filename}'")
        with open(filename, "w") as f:
            f.write(content)


jinja_environment = jinja2.Environment(
    loader=jinja2.FileSystemLoader(LOCAL_DIR / "templates")
)
