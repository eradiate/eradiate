from __future__ import annotations

from pathlib import Path

import attrs

import eradiate
from eradiate.plot import (
    absorption_database_spectral_coverage_ckd,
    absorption_database_spectral_coverage_mono,
)
from eradiate.radprops import CKDAbsorptionDatabase, MonoAbsorptionDatabase

from ..util import DOCS_ROOT_DIR, jinja_environment, savefig, write_if_modified


@attrs.define
class AbsorptionDatabaseInfo:
    keyword: str
    path: str
    spectral_sampling: str


def generate_absorption_database_visual(
    info: AbsorptionDatabaseInfo, outfile: Path, force=False
):
    eradiate.plot.set_style()

    # Create summary plot for an absorption database
    if outfile.is_file() and not force:  # Skip if file exists
        return

    print(f"Generating molecular absorption database visual in '{outfile}'")
    db = eradiate.radprops.AbsorptionDatabase.from_name(info.keyword)

    if isinstance(db, MonoAbsorptionDatabase):
        fig, _ = absorption_database_spectral_coverage_mono(db)
    elif isinstance(db, CKDAbsorptionDatabase):
        fig, _ = absorption_database_spectral_coverage_ckd(db)
    else:
        raise NotImplementedError

    savefig(fig, outfile, dpi=150, bbox_inches="tight")


def generate_summary():
    outfile_rst = DOCS_ROOT_DIR / "rst/data/absorption_databases.rst"
    outdir_visuals = DOCS_ROOT_DIR / "fig/absorption_databases"
    template = jinja_environment.get_template("absorption_databases.rst")

    absorption_databases = {"mono": [], "ckd": []}
    absorption_databases["mono"].extend(
        [
            AbsorptionDatabaseInfo(
                keyword="gecko",
                path="spectra/absorption/mono/gecko",
                spectral_sampling="0.01 cm⁻¹ in [250, 300] + [600, 3125] nm, 0.1 cm⁻¹ in [300, 600] nm",
            ),
            AbsorptionDatabaseInfo(
                keyword="komodo",
                path="spectra/absorption/mono/komodo",
                spectral_sampling="1 cm⁻¹",
            ),
        ]
    )
    absorption_databases["ckd"].extend(
        [
            AbsorptionDatabaseInfo(
                keyword="monotropa",
                path="spectra/absorption/ckd/monotropa",
                spectral_sampling="100 cm⁻¹",
            ),
            AbsorptionDatabaseInfo(
                keyword="mycena",
                path="spectra/absorption/ckd/mycena",
                spectral_sampling="10 nm",
            ),
            AbsorptionDatabaseInfo(
                keyword="panellus",
                path="spectra/absorption/ckd/panellus",
                spectral_sampling="1 nm",
            ),
        ]
    )

    for info in absorption_databases["mono"] + absorption_databases["ckd"]:
        outfile_visual = outdir_visuals / f"{info.keyword}.png"
        generate_absorption_database_visual(info, outfile_visual)

    result = template.render(absorption_databases=absorption_databases)
    write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate_summary()
