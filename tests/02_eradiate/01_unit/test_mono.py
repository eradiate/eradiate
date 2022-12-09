import numpy as np
import pytest

from eradiate.mono import WavelengthSet, included
from eradiate.scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum
from eradiate.units import unit_registry as ureg


def test_included():
    select = included(
        wmin=np.array([400.0, 500.0]) * ureg.nm,
        wmax=np.array([450.0, 550.0]) * ureg.nm,
    )
    selected = select(w=np.array([310, 410.0, 510.0, 610.0]) * ureg.nm)
    assert np.all(selected == np.array([False, True, True, False]))


def test_select_from_multiple_delta_spectrum():
    """
    Unit tests for :meth:`eradiate.mono.WavelengthSet.select_from_srf`.
    """

    wset = WavelengthSet(wavelengths=np.arange(280.0, 2400.0, 1.0) * ureg.nm)

    srf = MultiDeltaSpectrum(w=550.0 * ureg.nm)
    assert np.isclose(wset.select_with(srf).wavelengths, srf.w)

    srf = MultiDeltaSpectrum(w=555.5 * ureg.nm)
    assert np.isclose(wset.select_with(srf).wavelengths, srf.w)

    srf = MultiDeltaSpectrum(w=np.linspace(500.0, 600.0) * ureg.nm)
    assert np.allclose(wset.select_with(srf).wavelengths, srf.w)
    

@pytest.mark.parametrize(
    "srf, expected_wavelengths",
    [
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float),
            ),
            np.concatenate(
                [
                    np.arange(510.0, 521.0, 1.0),
                    np.arange(540.0, 571.0, 1.0),
                ]
            ) * ureg.nm
        ),
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.5, 600.5, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float),
            ),
            np.concatenate(
                [
                    np.arange(511.0, 521.0, 1.0),
                    np.arange(541.0, 571.0, 1.0),
                ]
            ) * ureg.nm
        ),
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.5, 600.5, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=float),
            ),
            np.concatenate(
                [
                    np.arange(511.0, 521.0, 1.0),
                    np.arange(541.0, 561.0, 1.0),
                    np.arange(581.0, 591.0, 1.0),
                ]
            ) * ureg.nm
        )

    ]
)
def test_select_from_interpolated_spectrum(srf, expected_wavelengths):
    """
    Unit tests for :meth:`eradiate.mono.WavelengthSet.select_from_srf`.
    """
    wset = WavelengthSet(wavelengths=np.arange(280.0, 2400.0, 1.0) * ureg.nm)
    new_wset = wset.select_with(srf)
    assert np.allclose(new_wset.wavelengths, expected_wavelengths)


def test_select_from_interpolated_spectrum_invalid():
    """
    Unit tests for :meth:`eradiate.mono.WavelengthSet.select_from_srf`.
    """
    wset = WavelengthSet(wavelengths=np.arange(280.0, 2400.0, 1.0) * ureg.nm)
    srf = InterpolatedSpectrum(
        wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
        values=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    )

    with pytest.raises(ValueError):
        wset.select_with(srf)