# This script walks through plugin files and extracts documentation that should
# go into the reference manual.

import os
import re
from pathlib import Path

# Section titles
SECTIONS = {
    "bsdfs": "BSDFs",
    "sensors": "Sensors",
}

# Plugin list
PLUGINS = {
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


def extract(in_filename, out_filename):
    with open(in_filename) as in_file, open(out_filename, "w") as out_file:
        inheader = False

        for line in in_file.readlines():
            match = re.match(r"^/\*\*! ?(.*)$", line)
            if match is not None:
                print(f"Processing {in_filename}")
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
-------

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

    for section, plugin_list in PLUGINS.items():
        # Create TOC pages
        section_rst = out_dir / f"toctrees/{section}.rst"
        os.makedirs(section_rst.parent, exist_ok=True)
        make_toc(section, SECTIONS[section], section_rst)

        # Extract plugin docs
        for plugin_name in plugin_list:
            plugin_cpp = plugin_dir / section / f"{plugin_name}.cpp"
            plugin_rst = out_dir / "plugins" / section / f"{plugin_name}.rst"
            os.makedirs(plugin_rst.parent, exist_ok=True)
            extract(plugin_cpp, plugin_rst)


if __name__ == "__main__":
    generate()
