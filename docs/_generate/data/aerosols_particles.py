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
    outfile_rst = DOCS_ROOT_DIR / "data/aerosols_particles.rst"
    outdir_visuals = DOCS_ROOT_DIR / "_images/particle_radprops"
    template = jinja_environment.get_template("aerosols_particles.rst")

    particle_radprops = []

    particle_radprops.extend(
        [
            ParticleRadpropsInfo(
                "govaerts_2021-continental-extrapolated",
                "aerosol/govaerts_2021-continental-extrapolated.nc",
                description="An aerosol dataset representative of continental "
                "aerosol classes to support the `RAMI4ATM benchmarking exercise "
                "<https://rami-benchmark.jrc.ec.europa.eu/_www/RAMI4ATM/"
                "phase_RAMI4ATM_p.php?strPhase=RAMI4ATM#tagATMcA>`__. "
                "The single-scattering properties were computed "
                "using miepython.",
            ),
            ParticleRadpropsInfo(
                "govaerts_2021-desert-extrapolated",
                "aerosol/govaerts_2021-desert-extrapolated.nc",
                description="An aerosol dataset representative of desert "
                "aerosol classes to support the `RAMI4ATM benchmarking exercise "
                "<https://rami-benchmark.jrc.ec.europa.eu/_www/RAMI4ATM/"
                "phase_RAMI4ATM_p.php?strPhase=RAMI4ATM#tagATMcA>`__. "
                "The single-scattering properties were computed "
                "using miepython.",
            ),
        ]
    )

    particle_radprops.extend(
        [
            ParticleRadpropsInfo(id, f"aerosol/{id}.nc")
            for id in [
                "sixsv-biomass_burning",
                "sixsv-continental",
                "sixsv-desert",
                "sixsv-maritime",
                "sixsv-stratospheric",
                "sixsv-urban",
            ]
        ]
    )
    for info in particle_radprops:
        outfile_visual = outdir_visuals / f"{info.keyword}.png"
        generate_particle_radprops_visual(info, outfile_visual)

    result = template.render(particle_radprops=particle_radprops).rstrip() + "\n"
    write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate_summary()
