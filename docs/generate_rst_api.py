import glob
import os
from importlib import import_module
from pathlib import Path
from textwrap import indent

import click
from rich.console import Console

console = Console()

# List of (module, variable) pairs
FACTORIES = [
    ("eradiate.radprops.rad_profile", "rad_profile_factory"),
    ("eradiate.scenes.atmosphere", "atmosphere_factory"),
    ("eradiate.scenes.atmosphere", "particle_distribution_factory"),
    ("eradiate.scenes.biosphere", "biosphere_factory"),
    ("eradiate.scenes.illumination", "illumination_factory"),
    ("eradiate.scenes.integrators", "integrator_factory"),
    ("eradiate.scenes.measure", "measure_factory"),
    ("eradiate.scenes.phase", "phase_function_factory"),
    ("eradiate.scenes.spectra", "spectrum_factory"),
    ("eradiate.scenes.surface", "surface_factory"),
]


def factory_data_docs(modname, varname, uline="="):
    """
    Return rst code for a factory instance located at modname.varname.
    """
    factory = getattr(import_module(modname), varname)
    fullname = f"{modname}.{varname}"
    underline = uline * len(fullname)

    table_header = ".. list-table::\n   :widths: 25 75"
    table_rows = "\n".join(
        [
            f"   * - ``{key}``\n     - :class:`{entry.cls.__name__}`"
            for key, entry in sorted(factory.registry.items())
        ]
    )
    table = "\n".join([table_header, "", table_rows])

    return f"""{fullname}
{underline}

.. data:: {modname}.{varname}
   :annotation: = {factory.__class__.__module__}.{factory.__class__.__qualname__}()

   Instance of :class:`{factory.__class__.__module__}.{factory.__class__.__qualname__}`

   .. rubric:: Registered types

{indent(table, "   ")}

""".lstrip()


def generate_factory_docs():
    """
    Generate rst documents to display factory documentation.
    """
    outdir = Path(__file__).parent.absolute() / "rst/reference/generated/factory"
    console.log(f"Generating factory docs in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    console.log(f"Cleaning up directory")
    files = glob.glob(str(outdir) + "/*.rst")
    for f in files:
        os.remove(f)

    for modname, varname in FACTORIES:
        outfname = outdir / f"{modname}.{varname}.rst"
        console.log(f"Writing {outfname.relative_to(outdir)}")

        with open(outfname, "w") as outfile:
            generated = factory_data_docs(modname, varname)
            # console.log(generated)
            outfile.write(generated)


@click.command()
def main():
    """
    Generate dynamic documentation parts.
    """
    generate_factory_docs()


if __name__ == "__main__":
    main()
