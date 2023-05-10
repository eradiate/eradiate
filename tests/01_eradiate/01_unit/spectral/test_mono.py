import numpy as np
import pytest

from eradiate.scenes.spectra import InterpolatedSpectrum, MultiDeltaSpectrum
from eradiate.spectral.mono import WavelengthSet
from eradiate.units import unit_registry as ureg


def test_select_with_multi_delta():
    """
    Unit tests for :meth:`.WavelengthSet.select_with_srf`.
    """

    wset = WavelengthSet(wavelengths=np.arange(280.0, 2400.0, 1.0) * ureg.nm)

    srf = MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm)
    assert np.isclose(srf.select_in(wset).wavelengths, srf.wavelengths)

    srf = MultiDeltaSpectrum(wavelengths=555.5 * ureg.nm)
    assert np.isclose(srf.select_in(wset).wavelengths, srf.wavelengths)

    srf = MultiDeltaSpectrum(wavelengths=np.linspace(500.0, 600.0) * ureg.nm)
    assert np.allclose(srf.select_in(wset).wavelengths, srf.wavelengths)


@pytest.mark.parametrize(
    "srf, expected_wavelengths",
    [
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float),
            ),
            np.arange(500.0, 581.0, 1.0) * ureg.nm,
        ),
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.5, 600.5, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float),
            ),
            np.arange(501.0, 581.0, 1.0) * ureg.nm,
        ),
        (
            InterpolatedSpectrum(
                wavelengths=np.linspace(500.0, 600.0, 11) * ureg.nm,
                values=np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float),
            ),
            np.concatenate(
                [
                    np.arange(500.0, 531.0, 1.0),
                    np.arange(540.0, 581.0, 1.0),
                ]
            )
            * ureg.nm,
        ),
    ],
)
def test_select_in_interpolated_spectrum(srf, expected_wavelengths):
    """
    Unit tests for :meth:`eradiate.scenes.spectra.InterpolatedSpectrum.select_in`.
    """
    wset = WavelengthSet(wavelengths=np.arange(280.0, 2400.0, 1.0) * ureg.nm)
    new_wset = srf.select_in(wset)
    assert np.allclose(new_wset.wavelengths, expected_wavelengths)
