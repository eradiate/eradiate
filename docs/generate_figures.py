# Render dot files
import subprocess
from pathlib import Path

DOT_FILES = [
    "fig/cuboid_leaf_cloud_params.dot",
]

for ext in ["svg", "png"]:
    for fname in DOT_FILES:
        fname = Path(fname)
        outfile = fname.parent / f"{fname.stem}.{ext}"
        cmd = ["dot", str(fname), f"-T{ext}", f"-o{outfile}"]
        print(" ".join(cmd))
        subprocess.call(args=cmd)
