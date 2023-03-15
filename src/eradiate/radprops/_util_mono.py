from __future__ import annotations

import pint

# This table hard-codes a wavenumber bin index for spectral data.
# Beware: this is fragile, it can break if not carefully maintained!
WAVENUMBER_BINS = {
    "spectra/absorption/us76_u86_4/us76_u86_4-spectra": [
        (4000, 5000),
        (5000, 6000),
        (6000, 7000),
        (7000, 8000),
        (8000, 9000),
        (9000, 10000),
        (10000, 11000),
        (11000, 12000),
        (12000, 13000),
        (13000, 14000),
        (14000, 15000),
        (15000, 16000),
        (16000, 17000),
        (17000, 18000),
        (18000, 19000),
        (19000, 20000),
        (20000, 21000),
        (21000, 22000),
        (22000, 23000),
        (23000, 24000),
        (24000, 25000),
        (25000, 25711),
    ]
}


def resolve_bin(group: str, wavenumber: pint.Quantity) -> tuple[int, int]:
    """
    Return the wavenumber bin corresponding to a group and a wavenumber value,
    if it exists.
    """
    wavenumber_m = wavenumber.m_as("cm^-1")

    for w_bin in WAVENUMBER_BINS[group]:
        w_min, w_max = w_bin
        if w_min <= wavenumber_m <= w_max:
            return w_bin

    raise ValueError(
        "Cannot find wavenumber bin corresponding to wavenumber "
        f"value {wavenumber} for group {group}"
    )


def get_us76_u86_4_spectrum_filename(wavelength: pint.Quantity):
    prefix = "spectra/absorption/us76_u86_4/us76_u86_4-spectra"
    wn_min, wn_max = resolve_bin(prefix, wavenumber=1.0 / wavelength)
    return f"{prefix}-{wn_min}_{wn_max}.nc"
