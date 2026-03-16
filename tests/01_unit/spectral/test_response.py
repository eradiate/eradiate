import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.spectral import SpectralResponseFunction
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF


class TestUniformSRF:
    def test_construct(self):
        assert UniformSRF() is not None

    def test_eval(self):
        srf = UniformSRF(400.0, 500.0, 1.0)
        value = srf.eval([300, 400, 450, 500, 600]).m
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(value, expected)


class TestDeltaSRF:
    @pytest.fixture(scope="class")
    def srf(self):
        yield DeltaSRF([440.0, 550.0, 660.0])

    def test_construct_from_float(self):
        srf = DeltaSRF(0.55 * ureg.micron)
        np.testing.assert_array_equal(srf.wavelengths.m_as("nm"), [550.0], strict=True)

    def test_construct_from_array(self):
        srf = DeltaSRF([440.0, 550.0, 660.0])
        np.testing.assert_array_equal(
            srf.wavelengths.m_as("nm"), [440.0, 550.0, 660.0], strict=True
        )

    def test_eval_single_value(self, srf):
        np.testing.assert_array_equal(
            srf.eval(500.0).m_as("dimensionless"), 0.0, strict=True
        )

    def test_eval_multiple_values(self, srf):
        np.testing.assert_array_equal(
            srf.eval([500.0]).m_as("dimensionless"), [0.0], strict=True
        )

        np.testing.assert_array_equal(
            srf.eval([440.0, 550.0, 660.0]).m_as("dimensionless"),
            [0.0, 0.0, 0.0],
            strict=True,
        )


class TestBandSRF:
    @pytest.fixture(scope="class")
    def srf(self):
        yield BandSRF(wavelengths=[500, 550, 600], values=[0, 1, 0])

    def test_construct(self):
        # Construct without leading and trailing zeros
        with pytest.warns(UserWarning):
            BandSRF(wavelengths=[500, 550, 600], values=[1, 1, 1])

        # Construct from ID
        srf = BandSRF.from_id("sentinel_2a-msi-3")
        np.testing.assert_array_equal(srf.support().m_as("nm"), [536.0, 583.0])

    def test_eval_single_value(self, srf):
        result = srf.eval(550.0).m_as("dimensionless")
        np.testing.assert_array_equal(result, 1.0, strict=True)

    def test_eval_multiple_values(self, srf):
        result = srf.eval([400, 500, 525, 550, 575, 600, 700]).m
        expected = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "wmin, wmax, expected",
        [(500, 600, 50.0), (500, 550, 25.0), (525, 575, 37.5)],
    )
    def test_integrate(self, srf, wmin, wmax, expected):
        result = srf.integrate(wmin, wmax).m_as("nm")
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "w, expected",
        [
            ([500, 550, 600], [25, 50]),
            ([525, 550, 575], [18.75, 37.5]),
            ([400, 500, 550, 600, 700], [0, 25, 50, 50]),
        ],
    )
    def test_integrate_cumulative(self, srf, w, expected):
        result = srf.integrate_cumulative(w).m_as("nm")
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "wavelengths, values, expected",
        [
            ([500, 550, 600], [0, 1, 0], 550.0),
            ([0.500, 0.533, 0.567, 0.600] * ureg.um, [0, 0.5, 1, 0], 555.666667),
        ],
    )
    def test_central_wavelength(self, wavelengths, values, expected):
        srf = BandSRF(wavelengths=wavelengths, values=values)
        result = srf.central_wavelength().m_as("nm")
        np.testing.assert_allclose(result, expected)

    def test_to_dataarray(self, srf):
        result = srf.to_dataarray()
        expected = xr.DataArray(srf.values.m, coords={"w": srf.wavelengths.m})
        xr.testing.assert_equal(result, expected)

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
            (
                550.0,
                1.0,
                3.0,
                None,
                False,
                [0.0625, 1.0, 0.0625],
                [549.0, 550.0, 551.0],
            ),
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
    def test_gaussian(self, wl_center, fwhm, cutoff, wl, pad, expected_srf, expected_w):
        if not pad:
            with pytest.warns(UserWarning):
                srf = BandSRF.gaussian(wl_center, fwhm, cutoff=cutoff, wl=wl, pad=pad)
        else:
            srf = BandSRF.gaussian(wl_center, fwhm, cutoff=cutoff, wl=wl, pad=pad)

        ds = srf.to_dataset()
        np.testing.assert_allclose(ds.srf.data, expected_srf, rtol=1e-5)
        np.testing.assert_allclose(ds.w.data, expected_w, rtol=1e-5)


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
