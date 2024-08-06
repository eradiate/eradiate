from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

import eradiate.data as data

from ..util import DOCS_ROOT_DIR, jinja_environment, savefig, write_if_modified

# ------------------------------------------------------------------------------
#                                   Constants
# ------------------------------------------------------------------------------


# Typical VIS/NIR domain split
DOMAINS = {"VIS": [0, 1000], "NIR": [1000, 2500]}

# List of instruments and bands
INSTRUMENTS = {
    "sentinel_2a-msi": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "8a",
        "9",
        "10",
        "11",
        "12",
    ],
    "sentinel_2b-msi": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "8a",
        "9",
        "10",
        "11",
        "12",
    ],
}

# ------------------------------------------------------------------------------
#                          Utility function definitions
# ------------------------------------------------------------------------------


def srf_meta(da):
    """
    Collect useful metadata about an SRF data array.
    """
    bands = da.band.values

    wmin = []
    wmax = []
    wcenter = []

    for i, _ in enumerate(bands):
        srf = da.isel(band=i).dropna("w")

        # Collect spectral bounds
        wmin.append(float(srf.w.min()))
        wmax.append(float(srf.w.max()))

        # Compute central wavelength
        wcenter.append((srf * srf.w).integrate("w") / srf.integrate("w"))

    wmin = xr.DataArray(data=wmin, coords={"band": ("band", da.band.data)}, name="wmin")
    wmax = xr.DataArray(data=wmax, coords={"band": ("band", da.band.data)}, name="wmax")
    wcenter = xr.DataArray(
        data=wcenter, coords={"band": ("band", da.band.data)}, name="wcenter"
    )

    return xr.Dataset({"wmin": wmin, "wmax": wmax, "wcenter": wcenter})


def split_domains(ds, domains=None):
    """
    Split bands of an instrument into pre-defined domains, based on
    the value of the central wavelength.
    """
    if domains is None:
        domains = DOMAINS

    meta = srf_meta(ds.srf)

    split = {}

    for domain, (domain_wmin, domain_wmax) in domains.items():
        selected_bands = (
            meta.where((meta.wcenter >= domain_wmin) & (meta.wcenter <= domain_wmax))
            .dropna("band")
            .band.values
        )
        split[domain] = selected_bands

    return split


def srf_ids(instrument_name, bands):
    return [f"{instrument_name}-{band}" for band in bands]


def srf_paths(instrument_name, bands):
    return [f"spectra/srf/{instrument_name}-{band}.nc" for band in bands]


def load_srfs(instrument_name, bands):
    fnames = srf_paths(instrument_name, bands)
    ds = xr.concat(
        [data.load_dataset(fname) for fname in fnames],
        pd.Index(bands, name="band"),
    )
    return ds


def plot_srfs(ds, domains=None):
    split = split_domains(ds, domains)

    nrows = len(split)

    fig, axs = plt.subplots(nrows, 1, layout="constrained")

    for irow, (domain, bands) in enumerate(split.items()):
        ax = axs[irow]
        ds.srf.sel(band=bands).plot(hue="band", ax=ax)
        sns.move_legend(
            ax, loc="center left", ncols=2, bbox_to_anchor=(1, 0.5), title="Band"
        )

        if irow != nrows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Wavelength [nm]")

        ax.set_ylabel("SRF [—]")
        # ax.set_title(domain)

    return fig, axs


# ------------------------------------------------------------------------------
#                          Plotting and rendering logic
# ------------------------------------------------------------------------------


@attrs.define
class InstrumentInfo:
    name: str
    bands: list
    ids: list
    paths: list


def generate_srf_visual(instrument: str, outfile: Path, force=False):
    # Create summary plot for an instrument
    if outfile.is_file() and not force:  # Skip if file exists
        return

    print(f"Generating SRF visual in '{outfile}'")
    bands = INSTRUMENTS[instrument]
    ds = load_srfs(instrument, bands)
    fig, _ = plot_srfs(ds)
    savefig(fig, outfile, dpi=150, bbox_inches="tight")


def generate_summary():
    outfile_rst = DOCS_ROOT_DIR / "rst/data/srf.rst"
    outdir_visuals = DOCS_ROOT_DIR / "fig/srf"
    template = jinja_environment.get_template("srf.rst")

    for instrument in INSTRUMENTS.keys():
        generate_srf_visual(instrument, outfile=outdir_visuals / f"{instrument}.png")

    instruments = [
        InstrumentInfo(
            name=name,
            bands=INSTRUMENTS[name],
            ids=srf_ids(name, INSTRUMENTS[name]),
            paths=srf_paths(name, INSTRUMENTS[name]),
        )
        for name in INSTRUMENTS.keys()
    ]

    result = template.render(instruments=instruments)
    write_if_modified(outfile_rst, result)


if __name__ == "__main__":
    generate_summary()
