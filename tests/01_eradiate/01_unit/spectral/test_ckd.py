import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.quad import Quad, QuadType
from eradiate.scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum
from eradiate.spectral import Bin, BinSet
from eradiate.spectral.ckd import (
    QuadSpec,
    QuadSpecErrorThreshold,
    QuadSpecFixed,
    QuadSpecMinError,
)


def test_ckd_bin():
    """
    Unit tests for :class:`eradiate.ckd.Bin`.
    """
    quad = Quad.gauss_legendre(16)

    # Construct a bin
    bin = Bin(wmin=500.0, wmax=510.0, quad=quad)

    # Scalar values are correctly converted to config default units
    assert bin.wmin == 500.0 * ureg.nm

    # Wrong bound ordering raises
    with pytest.raises(ValueError):
        Bin(wmin=510.0, wmax=500.0, quad=quad)


def test_select_with_multi_delta_1():
    """
    Unit tests for :meth:`.BinSet.select_with`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = MultiDeltaSpectrum(wavelengths=555.0 * ureg.nm)
    expected_selected_bins = [
        Bin(wmin=550.0 * ureg.nm, wmax=560.0 * ureg.nm),
    ]
    selected = srf.select_in(binset)
    for bin, expected_bin in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected_bin.wmin)


def test_select_with_multi_delta_2():
    """
    When wavelength fall between two bins, the left bin is selected by convention.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm)
    expected_selected_bins = [
        Bin(wmin=540.0 * ureg.nm, wmax=550.0 * ureg.nm),
    ]
    selected = srf.select_in(binset)
    for bin, expected_bin in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected_bin.wmin)


def test_select_from_connex_interpolated_spectrum_1():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.concatenate(
            [
                np.array([0.0]),
                np.ones(9),
                np.array([0.0]),
            ]
        )
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    wmin_expected = np.linspace(500.0, 600.0, 11) * ureg.nm
    assert len(selected.bins) == 10
    for bin, wmin in zip(selected.bins, wmin_expected):
        assert np.isclose(bin.wmin, wmin)
        assert np.isclose(bin.wmax, wmin + 10.0 * ureg.nm)


def test_select_from_connex_interpolated_spectrum_2():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    epsilon = np.finfo(float).eps

    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.concatenate(
            [
                np.array([epsilon]),
                np.ones(9),
                np.array([epsilon]),
            ]
        )
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    wmin_expected = np.linspace(490.0, 610.0, 13) * ureg.nm
    assert len(selected.bins) == 11
    for bin, wmin in zip(selected.bins, wmin_expected):
        assert np.isclose(bin.wmin, wmin)
        assert np.isclose(bin.wmax, wmin + 10.0 * ureg.nm)


def test_select_from_connex_interpolated_spectrum_3():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.concatenate(
            [
                np.array([0.0, 0.0]),
                np.ones(7),
                np.array([0.0, 0.0]),
            ]
        )
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    wmin_expected = np.linspace(510.0, 590.0, 9) * ureg.nm
    assert len(selected.bins) == 8
    for bin, wmin in zip(selected.bins, wmin_expected):
        assert np.isclose(bin.wmin, wmin)
        assert np.isclose(bin.wmax, wmin + 10.0 * ureg.nm)


def test_select_from_non_connex_interpolated_spectrum_1():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    expected_selected_bins = [
        Bin(wmin=510.0 * ureg.nm, wmax=520.0 * ureg.nm),
        Bin(wmin=520.0 * ureg.nm, wmax=530.0 * ureg.nm),
        Bin(wmin=530.0 * ureg.nm, wmax=540.0 * ureg.nm),
        Bin(wmin=550.0 * ureg.nm, wmax=560.0 * ureg.nm),
        Bin(wmin=560.0 * ureg.nm, wmax=570.0 * ureg.nm),
        Bin(wmin=570.0 * ureg.nm, wmax=580.0 * ureg.nm),
        Bin(wmin=580.0 * ureg.nm, wmax=590.0 * ureg.nm),
    ]
    for bin, expected in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected.wmin)
        assert np.isclose(bin.wmax, expected.wmax)


def test_select_from_non_connex_interpolated_spectrum_2():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )
    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    expected_selected_bins = [
        Bin(wmin=510.0 * ureg.nm, wmax=520.0 * ureg.nm),
        Bin(wmin=520.0 * ureg.nm, wmax=530.0 * ureg.nm),
        Bin(wmin=550.0 * ureg.nm, wmax=560.0 * ureg.nm),
        Bin(wmin=560.0 * ureg.nm, wmax=570.0 * ureg.nm),
        Bin(wmin=570.0 * ureg.nm, wmax=580.0 * ureg.nm),
        Bin(wmin=580.0 * ureg.nm, wmax=590.0 * ureg.nm),
        Bin(wmin=590.0 * ureg.nm, wmax=600.0 * ureg.nm),
    ]
    for bin, expected in zip(selected.bins, expected_selected_bins):
        assert np.isclose(bin.wmin, expected.wmin)
        assert np.isclose(bin.wmax, expected.wmax)


def test_select_from_connex_interpolated_spectrum_misaligned():
    """
    Unit tests for :meth:`.BinSet.select_from_srf`.
    """
    binset = BinSet.arange(
        start=280.0 * ureg.nm,
        stop=2400.0 * ureg.nm,
        step=10.0 * ureg.nm,
    )

    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(505.0, 605.0, 11) * ureg.nm,
        values=np.concatenate(
            [
                np.array([0.0, 0.0]),
                np.ones(7),
                np.array([0.0, 0.0]),
            ]
        )
        * ureg.dimensionless,
    )

    selected = srf.select_in(binset)
    wmin_expected = np.linspace(510.0, 600.0, 10) * ureg.nm
    assert len(selected.bins) == 9
    for bin, wmin in zip(selected.bins, wmin_expected):
        assert np.isclose(bin.wmin, wmin)
        assert np.isclose(bin.wmax, wmin + 10.0 * ureg.nm)


def test_quad_spec_default():
    assert QuadSpec.default() == QuadSpecFixed(quad_type=QuadType.GAUSS_LEGENDRE, n=16)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            {"type": "fixed", "n": 8, "quad_type": "gauss_lobatto"},
            QuadSpecFixed(8, "gauss_lobatto"),
        ),
        (
            {"type": "minimize", "nmax": 8},
            QuadSpecMinError(8),
        ),
        (
            {"type": "threshold", "nmax": 8, "threshold": 0.01},
            QuadSpecErrorThreshold(0.01, 8),
        ),
        ({"type": "invalid"}, ValueError),
        ({}, ValueError),
    ],
)
def test_quad_spec_from_dict(value, expected):
    """
    Unit tests for :class:`.QuadSpec`.
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            QuadSpec.from_dict(value)

    else:
        quad_spec = QuadSpec.from_dict(value)
        assert quad_spec == expected
