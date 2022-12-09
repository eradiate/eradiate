import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.ckd import Bin, BinSet, includes, overlaps
from eradiate.quad import Quad
from eradiate.scenes.spectra import InterpolatedSpectrum


def test_ckd_bin():
    """
    Unit tests for :class:`eradiate.ckd.Bin`.
    """
    quad = Quad.gauss_legendre(16)

    # Construct a bin
    bin = Bin(wmin=500.0, wmax=510.0, quad=quad)

    # Scalar values are correctly converted to config default units
    assert bin.wmin == 500.0 * ureg.nm
    assert bin.wmax == 510.0 * ureg.nm

    # Central wavelength is the mean of bounds
    assert bin.wcenter == 505.0 * ureg.nm

    # Width is the difference between bounds
    assert bin.width == 10.0 * ureg.nm

    # Wrong bound ordering raises
    with pytest.raises(ValueError):
        Bin(wmin=510.0, wmax=500.0, quad=quad)


@pytest.mark.parametrize(("bin", "wmin", "wmax", "selected"), [
    (Bin(wmin=500.0, wmax=510.0), 505.0 * ureg.nm, 515.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 500.0 * ureg.nm, 510.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 500.0 * ureg.nm, 505.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 505.0 * ureg.nm, 510.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 495.0 * ureg.nm, 505.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 505.0 * ureg.nm, 515.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 490.0 * ureg.nm, 500.0 * ureg.nm, False),
    (Bin(wmin=500.0, wmax=510.0), 510.0 * ureg.nm, 520.0 * ureg.nm, False),
    (Bin(wmin=500.0, wmax=510.0), 400.0 * ureg.nm, 490.0 * ureg.nm, False),
    (Bin(wmin=500.0, wmax=510.0), 520.0 * ureg.nm, 530.0 * ureg.nm, False),
])
def test_overlaps(bin, wmin, wmax, selected):
    select = overlaps(wmin, wmax)
    assert select(bin) == selected

@pytest.mark.parametrize(("bin", "w", "selected"), [
    (Bin(wmin=500.0, wmax=510.0), 505.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 500.0 * ureg.nm, True),
    (Bin(wmin=500.0, wmax=510.0), 510.0 * ureg.nm, False),
    (Bin(wmin=500.0, wmax=510.0), 495.0 * ureg.nm, False),
    (Bin(wmin=500.0, wmax=510.0), 515.0 * ureg.nm, False),
])
def test_includes(bin, w, selected):
    select = includes(w)
    assert select(bin) == selected
    

def test_select_from_connex_interpolated_spectrum():
    """
    Unit tests for :meth:`eradiate.ckd.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )
    
    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.ones(11),
    )

    selected = binset.select_with(srf)
    wmin_expected = np.linspace(500.0, 600.0, 11) * ureg.nm
    assert len(selected.bins) == 10
    for bin, wmin in zip(selected.bins, wmin_expected):
        assert np.isclose(bin.wmin, wmin)
        assert np.isclose(bin.wmax, wmin + 10.0 * ureg.nm)


def test_select_from_non_connex_interpolated_spectrum_1():
    """
    Unit tests for :meth:`eradiate.ckd.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )
    
    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    selected = binset.select_with(srf)
    expected_selected_bins = [
        Bin(wmin=520.0 * ureg.nm, wmax=530.0 * ureg.nm),
        Bin(wmin=560.0 * ureg.nm, wmax=570.0 * ureg.nm),
        Bin(wmin=570.0 * ureg.nm, wmax=580.0 * ureg.nm),
        Bin(wmin=580.0 * ureg.nm, wmax=590.0 * ureg.nm),
    ]
    for bin, expected in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected.wmin)
        assert np.isclose(bin.wmax, expected.wmax)

def test_select_from_non_connex_interpolated_spectrum_2():
    """
    Unit tests for :meth:`eradiate.ckd.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )
    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    selected = binset.select_with(srf)
    expected_selected_bins = [
        Bin(wmin=520.0 * ureg.nm, wmax=530.0 * ureg.nm),
        Bin(wmin=560.0 * ureg.nm, wmax=570.0 * ureg.nm),
        Bin(wmin=570.0 * ureg.nm, wmax=580.0 * ureg.nm),
        Bin(wmin=580.0 * ureg.nm, wmax=590.0 * ureg.nm),
    ]
    for bin, expected in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected.wmin)
        assert np.isclose(bin.wmax, expected.wmax)
