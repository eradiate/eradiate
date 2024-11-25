import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.spectral import SpectralResponseFunction
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF, make_gaussian


def test_uniform_srf():
    # Default constructor succeeds
    assert UniformSRF() is not None

    # Evaluation works as expected
    srf = UniformSRF(400.0, 500.0, 1.0)
    value = srf.eval([300, 400, 450, 500, 600]).m
    expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(value, expected)


def test_delta_srf():
    # Construct from float
    srf = DeltaSRF(0.55 * ureg.micron)
    np.testing.assert_array_equal(
        srf.wavelengths.m_as("nm"),
        [550.0],
        strict=True,
    )

    # Construct from array
    srf = DeltaSRF([440.0, 550.0, 660.0])
    np.testing.assert_array_equal(
        srf.wavelengths.m_as("nm"),
        [440.0, 550.0, 660.0],
        strict=True,
    )

    # Evaluate with single value
    np.testing.assert_array_equal(
        srf.eval(500.0).m_as("dimensionless"),
        0.0,
        strict=True,
    )

    # Evaluate with multiple values
    np.testing.assert_array_equal(
        srf.eval([500.0]).m_as("dimensionless"),
        [0.0],
        strict=True,
    )

    np.testing.assert_array_equal(
        srf.eval([440.0, 550.0, 660.0]).m_as("dimensionless"),
        [0.0, 0.0, 0.0],
        strict=True,
    )


def test_band_srf():
    # Construct
    srf = BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])

    # Construct without leading and trailing zeros
    with pytest.warns(UserWarning):
        BandSRF(wavelengths=[500, 550, 600], values=[1, 1, 1])

    # Evaluate with single value
    np.testing.assert_array_equal(
        srf.eval(550.0).m_as("dimensionless"), 1.0, strict=True
    )

    # Evaluate with multiple values
    values = srf.eval([400, 500, 525, 550, 575, 600, 700]).m
    expected = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    np.testing.assert_array_equal(values, expected)

    # Construct from ID
    srf = BandSRF.from_id("sentinel_2a-msi-3")
    np.testing.assert_array_equal(srf.support().m_as("nm"), [536.0, 583.0])

    # Integrate
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 600)
        .m_as("nm"),
        50.0,
    )
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 550)
        .m_as("nm"),
        25.0,
    )
    np.testing.assert_equal(
        BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])
        .integrate(500, 550)
        .m_as("nm"),
        25.0,
    )

    # Integrate (cumulative)
    integral = BandSRF(
        wavelengths=[500, 550, 600], values=[0, 1, 0]
    ).integrate_cumulative([500, 550, 600])
    np.testing.assert_equal(integral.m_as("nm"), [25, 50])

    integral = BandSRF(
        wavelengths=[500, 550, 600], values=[0, 1, 0]
    ).integrate_cumulative([525, 550, 575])
    np.testing.assert_equal(integral.m_as("nm"), [18.75, 37.5])

    integral = BandSRF(
        wavelengths=[500, 550, 600], values=[0, 1, 0]
    ).integrate_cumulative([400, 500, 550, 600, 700])
    np.testing.assert_equal(integral.m_as("nm"), [0, 25, 50, 50])

    # Export to xarray
    da = BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0]).to_dataarray()
    expected = xr.DataArray(np.array([0, 1, 0]), coords={"w": [500, 550, 600]})
    assert xr.testing.assert_equal(da, expected)


@pytest.mark.parametrize(
    "value, expected",
    [
        (DeltaSRF([550]), DeltaSRF),
        ({"type": "delta", "wavelengths": [550]}, DeltaSRF),
        ({"type": "uniform", "wmin": 500, "wmax": 600}, UniformSRF),
        ("sentinel_2a-msi-3", BandSRF),
    ],
    ids=["instance", "dict_uniform", "dict_delta", "str_band"],
)
def test_convert(value, expected):
    converted = SpectralResponseFunction.convert(value)
    assert isinstance(converted, expected)


@pytest.mark.parametrize(
    "wl_center, fwhm, cutoff, wl, pad, expected_srf, expected_w",
    [
        (
            0.0,
            1.0,
            3.0,
            np.linspace(-5, 5, 11),
            False,
            [0.0625, 1.0, 0.0625],
            [-1.0, 0.0, 1.0],
        ),
        (
            0.0,
            3.0,
            3.0,
            np.linspace(-5, 5, 11),
            False,
            [0.0625, 0.291632, 0.734867, 1.0, 0.734867, 0.291632, 0.0625],
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        ),
        (
            0.0,
            1.0,
            3.0,
            np.linspace(-1, 1, 11),
            False,
            [
                0.0625,
                0.169576,
                0.368567,
                0.641713,
                0.895025,
                1.0,
                0.895025,
                0.641713,
                0.368567,
                0.169576,
                0.0625,
            ],
            np.linspace(-1, 1, 11),
        ),
        (550.0, 1.0, 3.0, None, False, [0.0625, 1.0, 0.0625], [549.0, 550.0, 551.0]),
        (
            550.0,
            1.0,
            3.0,
            None,
            True,
            [0.0, 0.0625, 1.0, 0.0625, 0.0],
            [548.0, 549.0, 550.0, 551.0, 552.0],
        ),
    ],
)
def test_make_gaussian(wl_center, fwhm, cutoff, wl, pad, expected_srf, expected_w):
    ds = make_gaussian(wl_center, fwhm, cutoff=cutoff, wl=wl, pad=pad)
    np.testing.assert_allclose(ds.srf.data, expected_srf, rtol=1e-5)
    np.testing.assert_allclose(ds.w.data, expected_w, rtol=1e-5)
