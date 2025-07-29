# This script walks through plugin files and extracts documentation that should
# go into the reference manual.
import glob
import re
import typing as t
from pathlib import Path
from textwrap import dedent

# Section titles (order matters)
SECTIONS = {
    "shapes": "Shapes",
    "bsdfs": "BSDFs",
    "media": "Media",
    "phase": "Phase functions",
    "emitters": "Emitters",
    "sensors": "Sensors",
    "textures": "Textures",
    "spectra": "Spectra",
    "volumes": "Volume data sources",
    "integrators": "Integrators",
    "samplers": "Samplers",
    "films": "Films",
    "rfilters": "Reconstruction filters",
}

# Plugin list
PLUGIN_ORDERS = {
    "bsdfs": [
        "bilambertian",
        "rpv",
    ],
    "sensors": [
        "distantflux",
        "hdistant",
        "mdistant",
        "mradiancemeter",
    ],
    "emitters": [
        "astroobject",
    ],
}


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


def underline(s: str, c: str = "-"):
    return c * len(s)


def extract_plugin_docs(filename: Path) -> t.List[str]:
    result = []

    with open(filename) as in_file:
        inheader = False

        for line in in_file.readlines():
            match = re.match(r"^/\*\*! ?(.*)$", line)
            if match is not None:
                line = match.group(1).replace("%", "\%")
                result.append(line.rstrip())
                inheader = True
                continue

            if not inheader:
                continue

            if re.search(r"^[\s\*]*\*/$", line):
                inheader = False
                continue

            result.append(line.rstrip())

    return ("\n".join(result)).lstrip()


def make_section_title(section, title) -> str:
    return dedent(rf"""
        .. _sec-reference_plugins-{section}:

        {title}
        {underline(title, "=")}
    """).strip()


def generate():
    root_dir = (Path(__file__) / "../../..").resolve()
    plugin_dir = root_dir / "ext/mitsuba/src/eradiate_plugins"
    print(f"Looking for plugins in {plugin_dir}")
    out_dir = root_dir / "docs/reference_plugins/generated"
    print(f"Generating plugin docs in '{out_dir}'")

    for section, section_title in SECTIONS.items():
        try:
            plugin_order = PLUGIN_ORDERS[section]
        except KeyError:
            plugin_order = []

        # Get plugin filenames
        section_dir = plugin_dir / section
        plugin_list = [
            Path(filename).stem for filename in glob.glob(f"{section_dir}/*.cpp")
        ]

        if not plugin_list:
            print(f"No plugin in section '{section}'")
            continue

        # Sort plugin list according to the ordering dict
        def index(x):
            try:
                return plugin_order.index(x)
            except ValueError:
                return 1000

        plugin_list.sort(key=index)

        # Extract plugin docs
        section_rst = out_dir / f"{section}.rst"
        section_content = [make_section_title(section, section_title)]

        for plugin_name in plugin_list:
            print(f"Extracting plugin docs from '{plugin_dir}' to '{out_dir}'")
            plugin_cpp = plugin_dir / section / f"{plugin_name}.cpp"
            section_content.append(extract_plugin_docs(plugin_cpp))

        write_if_modified(section_rst, "\n\n".join(section_content))


if __name__ == "__main__":
    generate()
