import pytest
from pinttr.exceptions import UnitsError

from eradiate import unit_registry as ureg
from eradiate.scenes.spectra import UniformSpectrum, spectrum_factory


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            {"type": "uniform", "value": 1.0},
            UniformSpectrum(quantity="radiance", value=1.0),
        ),
        (
            1.0,
            UniformSpectrum(quantity="radiance", value=1.0),
        ),
        (
            ureg.Quantity(1e6, "W/km^2/sr/nm"),
            UniformSpectrum(quantity="radiance", value=1.0),
        ),
        (
            1,
            UniformSpectrum(quantity="radiance", value=1.0),
        ),
        (
            ureg.Quantity(1, "W/m^2/nm"),
            UnitsError,
        ),
    ],
    ids=["dict", "float", "quantity", "int", "wrong_units"],
)
def test_converter(mode_mono, tested, expected):
    """
    Tests for the Spectrum factory's conversion protocol.
    """

    if isinstance(expected, UniformSpectrum):
        s = spectrum_factory.converter("radiance")(tested)
        assert (s.quantity == expected.quantity) and (
            s.value == expected.value
        ), f"Failed: converted value is {s}"

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            spectrum_factory.converter("radiance")(tested)

    else:
        raise RuntimeError
