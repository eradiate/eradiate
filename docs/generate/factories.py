from __future__ import annotations

from importlib import import_module

import attrs

from .util import DOCS_ROOT_DIR, jinja_environment, write_if_modified


@attrs.define
class FactoryInfo:
    @attrs.define
    class RegisteredType:
        keyword: str
        cls: str

    qualified_name: str
    parent_type: str
    registered_types: list[RegisteredType]


# List of (module, variable) pairs
FACTORIES = [
    ("eradiate.radprops", "rad_profile_factory"),
    ("eradiate.scenes.atmosphere", "atmosphere_factory"),
    ("eradiate.scenes.atmosphere", "particle_distribution_factory"),
    ("eradiate.scenes.biosphere", "biosphere_factory"),
    ("eradiate.scenes.bsdfs", "bsdf_factory"),
    ("eradiate.scenes.illumination", "illumination_factory"),
    ("eradiate.scenes.integrators", "integrator_factory"),
    ("eradiate.scenes.measure", "measure_factory"),
    ("eradiate.scenes.phase", "phase_function_factory"),
    ("eradiate.scenes.shapes", "shape_factory"),
    ("eradiate.scenes.spectra", "spectrum_factory"),
    ("eradiate.scenes.surface", "surface_factory"),
]

FACTORY_INFOS = []

for modname, varname in FACTORIES:
    factory = getattr(import_module(modname), varname)
    qualified_name = f"{modname}.{varname}"
    parent_type = f"{factory.__class__.__module__}.{factory.__class__.__qualname__}"
    registered_types = [
        FactoryInfo.RegisteredType(
            keyword=key,
            cls=f"{factory.get_type(key).__name__}",
        )
        for key in sorted(factory.registry.keys())
    ]
    FACTORY_INFOS.append(
        FactoryInfo(
            qualified_name=qualified_name,
            parent_type=parent_type,
            registered_types=registered_types,
        )
    )


def generate():
    outdir = DOCS_ROOT_DIR / "rst/reference_api/generated/factory"
    print(f"Generating factory docs in '{outdir}'")
    template = jinja_environment.get_template("factory.rst")

    for entry in FACTORY_INFOS:
        outfile_rst = outdir / f"{entry.qualified_name}.rst"
        result = template.render(entry=entry)
        write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate()
