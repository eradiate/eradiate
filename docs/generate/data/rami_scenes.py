import itertools

from ..util import DOCS_ROOT_DIR, jinja_environment, write_if_modified

RAMI_SCENE_COMMENTS = {
    "HET51_WWO_TLS": (
        "This version of the Wytham Wood scene uses data from the updated v2 "
        "dataset."
    )
}


def generate_summary():
    from eradiate.scenes.biosphere import (
        RAMIActualCanopies,
        RAMIHeterogeneousAbstractCanopies,
        RAMIHomogeneousAbstractCanopies,
    )

    outfile_rst = DOCS_ROOT_DIR / "rst/data/rami_scenes.rst"
    template = jinja_environment.get_template("rami_scenes.rst")

    scenes = [
        {
            "rami_id": x.value,
            "description": x.name.replace("_", " ").lower(),
            "comments": RAMI_SCENE_COMMENTS.get(x.value, ""),
            "image_file": f"{x.name}_30_90.png",
        }
        for x in itertools.chain(
            RAMIHomogeneousAbstractCanopies,
            RAMIHeterogeneousAbstractCanopies,
            RAMIActualCanopies,
        )
    ]

    result = template.render(scenes=scenes)
    write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate_summary()
