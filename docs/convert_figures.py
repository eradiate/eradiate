"""
Convert SVG figures to PNG. Requires Inkscape 1.0 (with new CLI).
"""

import subprocess
from pathlib import Path

FIG_PATH = Path(__file__).parent / "fig"


def svg_to_png():
    svg_files = list(FIG_PATH.glob("*.svg"))
    for file in svg_files:
        cmd = [
            "inkscape",
            "--export-type=png",
            "--export-dpi=96",
            str(file.absolute()),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    svg_to_png()
