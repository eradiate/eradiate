"""
Unit tests for the MultiDeltaSpectrum class.
"""

import numpy as np
import pint
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import MultiDeltaSpectrum
from eradiate.spectral import BinSet


@pytest.mark.parametrize(
    "tested, expected",
    [
        ({}, TypeError),
        (
            {"wavelengths": [550.0, 660.0] * ureg.nm},
            MultiDeltaSpectrum(wavelengths=[550.0, 660.0] * ureg.nm),
        ),
        (
            {"wavelengths": [550.0, 660.0]},
            MultiDeltaSpectrum(wavelengths=[550.0, 660.0] * ureg.nm),
        ),
        (
            {"wavelengths": 550.0 * ureg.nm},
            MultiDeltaSpectrum(wavelengths=[550.0] * ureg.nm),
        ),
        ({"wavelengths": 550.0}, MultiDeltaSpectrum(wavelengths=[550.0] * ureg.nm)),
        ({"wavelengths": [550.0, -550.0]}, ValueError),
        (
            {"wavelengths": [660.0, 550.0]},
            MultiDeltaSpectrum(wavelengths=[550.0, 660.0] * ureg.nm),
        ),
        (
            {"wavelengths": [660.0, 550.0, 550.0]},
            MultiDeltaSpectrum(wavelengths=[550.0, 660.0] * ureg.nm),
        ),
    ],
    ids=[
        "no_args",
        "array_arg",
        "unitless_array_arg",
        "scalar_arg",
        "unitless_scalar_arg",
        "negative_values",
        "unsorted_values",
        "duplicate_values",
    ],
)
def test_multi_delta_construct(modes_all, tested, expected):
    if isinstance(expected, MultiDeltaSpectrum):
        s = MultiDeltaSpectrum(**tested)
        assert isinstance(s.wavelengths, pint.Quantity)
        assert np.allclose(s.wavelengths, expected.wavelengths)

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            MultiDeltaSpectrum(**tested)

    else:
        raise RuntimeError


@pytest.mark.parametrize(
    "multi_delta_wavelengths, expected_selected_wcenters",
    [
        ([555.0] * ureg.nm, [555.0] * ureg.nm),
        ([505.0, 595.0] * ureg.nm, [505.0, 595.0] * ureg.nm),
        (
            np.linspace(500.0, 600.0, 100) * ureg.nm,
            np.arange(505.0, 600.0, 10.0) * ureg.nm,
        ),
        ([510.0] * ureg.nm, [505.0] * ureg.nm),
    ],
    ids=[
        "single_value",
        "multiple_value",
        # If more values than bins are passed, still works as expected
        "many_values",
        # When wavelength falls between two bins, left bin is selected by convention
        "priority_left",
    ],
)
def test_multi_delta_select_ckd(multi_delta_wavelengths, expected_selected_wcenters):
    binset = BinSet.arange(start=500.0, stop=600.0, step=10.0)
    srf = MultiDeltaSpectrum(wavelengths=multi_delta_wavelengths)
    selected = srf.select_in(binset)
    assert np.allclose(selected.wcenters, expected_selected_wcenters)
