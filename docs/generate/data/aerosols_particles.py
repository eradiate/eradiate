from __future__ import annotations

from pathlib import Path

import attrs

import eradiate
from eradiate.plot import dashboard_particle_dataset

from ..util import DOCS_ROOT_DIR, jinja_environment, savefig, write_if_modified


@attrs.define
class ParticleRadpropsInfo:
    keyword: str
    fname: str
    description: str | None = attrs.field(default=None)
    aliases: list[str] = attrs.field(factory=list)


def generate_particle_radprops_visual(
    info: ParticleRadpropsInfo, outfile: Path, force=False
):
    eradiate.plot.set_style()

    # Create dashboards for a dataset
    if outfile.is_file() and not force:  # Skip if file exists
        return

    with eradiate.data.open_dataset(info.fname) as ds:
        print(f"Generating particle radiative property visual in '{outfile}'")
        fig, _ = dashboard_particle_dataset(ds)
        savefig(fig, outfile, dpi=150, bbox_inches="tight")


def generate_summary():
    outfile_rst = DOCS_ROOT_DIR / "rst/data/aerosols_particles.rst"
    outdir_visuals = DOCS_ROOT_DIR / "fig/particle_radprops"
    template = jinja_environment.get_template("aerosols_particles.rst")

    particle_radprops = [
        ParticleRadpropsInfo(id, f"spectra/particles/{id}.nc")
        for id in [
            "govaerts_2021-continental-extrapolated",
            "govaerts_2021-desert-extrapolated",
            "sixsv-biomass_burning",
            "sixsv-continental",
            "sixsv-desert",
            "sixsv-maritime",
            "sixsv-stratospheric",
            "sixsv-urban",
        ]
    ]
    for info in particle_radprops:
        outfile_visual = outdir_visuals / f"{info.keyword}.png"
        generate_particle_radprops_visual(info, outfile_visual)

    result = template.render(particle_radprops=particle_radprops)
    write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate_summary()
