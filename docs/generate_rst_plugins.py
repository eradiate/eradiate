# This script walks through plugin files and extracts documentation that should
# go into the reference manual.
import glob
import os
import re
from pathlib import Path

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
}


def underline(s: str, c: str = "-"):
    return c * len(s)


def extract(in_filename: Path, out_filename: Path):
    with open(in_filename) as in_file, open(out_filename, "w") as out_file:
        inheader = False

        for line in in_file.readlines():
            match = re.match(r"^/\*\*! ?(.*)$", line)
            if match is not None:
                line = match.group(1).replace("%", "\%")
                out_file.write(line + "\n")
                inheader = True
                continue

            if not inheader:
                continue

            if re.search(r"^[\s\*]*\*/$", line):
                inheader = False
                continue

            out_file.write(line)


def make_toc(section, title, out_filename):
    with open(out_filename, "w") as out_file:
        out_file.write(
            rf"""
.. _sec-reference_plugins-{section}:

{title}
{underline(title)}

.. toctree::
   :maxdepth: 1
   :glob:
   
   ../plugins/{section}/*
"""
        )


def generate():
    root_dir = Path(__file__).absolute().parent.parent
    plugin_dir = root_dir / "src/plugins/src"
    out_dir = root_dir / "docs/rst/reference_plugins/generated"

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

        # Create TOC page
        section_rst = out_dir / f"toctrees/{section}.rst"
        os.makedirs(section_rst.parent, exist_ok=True)
        make_toc(section, section_title, section_rst)

        # Sort plugin list according to the ordering dict
        def index(x):
            try:
                return plugin_order.index(x)
            except ValueError:
                return 1000

        plugin_list.sort(key=index)

        # Extract plugin docs
        for plugin_name in plugin_list:
            plugin_cpp = plugin_dir / section / f"{plugin_name}.cpp"
            plugin_rst = out_dir / "plugins" / section / f"{plugin_name}.rst"
            os.makedirs(plugin_rst.parent, exist_ok=True)
            print(f"Processing {Path(plugin_cpp).relative_to(root_dir)}")
            extract(plugin_cpp, plugin_rst)


if __name__ == "__main__":
    generate()
