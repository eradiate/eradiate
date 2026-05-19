"""Unit tests for eradiate.data.convert."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.data.convert import aer_v1_to_aer_core_v2, libradtran_to_aer_core_v2, make_aer_core_v2


def _make_aer_v1(mu_values: np.ndarray) -> xr.Dataset:
    """
    Build a minimal synthetic Aer v1 dataset with the given mu grid.

    The phase values are wavelength-independent and equal to 1/(4π) sr^-1
    (isotropic) for convenience.  Only the (i=0, j=0) component is included.
    """

    nw = 3
    nmu = len(mu_values)
    w = np.array([400.0, 550.0, 700.0])

    return xr.Dataset(
        data_vars={
            "sigma_t": ("w", np.ones(nw), {"units": "1/km"}),
            "albedo": ("w", 0.9 * np.ones(nw), {"units": ""}),
            "phase": (
                ("w", "mu", "i", "j"),
                np.ones((nw, nmu, 1, 1)) / (4.0 * np.pi),
                {"units": "sr^-1"},
            ),
        },
        coords={
            "w": ("w", w, {"units": "nm"}),
            "mu": ("mu", mu_values, {"units": ""}),
            "i": ("i", np.array([0], dtype=np.int64), {"units": ""}),
            "j": ("j", np.array([0], dtype=np.int64), {"units": ""}),
        },
        attrs={"history": "synthetic test dataset"},
    )


def _make_libradtran_ds(ntheta_per_w: list[int]) -> xr.Dataset:
    """
    Build a minimal synthetic dataset in libRadtran format.

    Parameters
    ----------
    ntheta_per_w
        Number of valid theta samples per wavelength. ``nthetamax`` is set to
        the maximum.
    """
    nw = len(ntheta_per_w)
    nthetamax = max(ntheta_per_w)
    nphamat = 1
    nmommax = 4

    wavelen = np.linspace(400.0, 700.0, nw)
    ext_vals = np.ones(nw)
    ssa_vals = 0.9 * np.ones(nw)

    # ntheta: (nlam, nphamat)
    ntheta = np.array([[n] for n in ntheta_per_w], dtype=np.int32)

    # theta: (nlam, nphamat, nthetamax) — NaN-padded beyond ntheta
    theta_arr = np.full((nw, nphamat, nthetamax), np.nan)
    phase_arr = np.full((nw, nphamat, nthetamax), np.nan)
    for iw, n in enumerate(ntheta_per_w):
        angles = np.linspace(0.0, 180.0, n)
        theta_arr[iw, 0, :n] = angles
        phase_arr[iw, 0, :n] = 1.0 / (4.0 * np.pi)  # isotropic

    # pmom: (nlam, nphamat, nmommax)
    pmom_arr = np.full((nw, nphamat, nmommax), np.nan)
    pmom_arr[:, 0, 0] = 1.0

    return xr.Dataset(
        data_vars={
            "wavelen": ("nlam", wavelen, {"units": "nm"}),
            "ext": ("nlam", ext_vals, {"units": "1/km"}),
            "ssa": ("nlam", ssa_vals, {"units": ""}),
            "ntheta": (("nlam", "nphamat"), ntheta),
            "theta": (("nlam", "nphamat", "nthetamax"), theta_arr),
            "phase": (("nlam", "nphamat", "nthetamax"), phase_arr),
            "pmom": (("nlam", "nphamat", "nmommax"), pmom_arr),
        }
    )


class TestMakeAerCoreV2:
    class TestCheck:
        """Tests for the mu sort-order check in make_aer_core_v2()."""

        def _build_args(self, mu_1d: np.ndarray):
            """Return keyword args for make_aer_core_v2 given a 1-D mu array."""
            nw = 2
            nangle = len(mu_1d)
            mu_2d = np.broadcast_to(mu_1d, (nw, nangle)).copy() * ureg.dimensionless
            theta_2d = (np.arccos(mu_2d.m) * ureg.rad).to("deg")
            phase = np.ones((1, nw, nangle)) * ureg("1/sr")
            return {
                "w": np.array([400.0, 700.0]) * ureg.nm,
                "phamat": ["11"],
                "mu": mu_2d,
                "theta": theta_2d,
                "ext": np.ones(nw) * ureg("1/km"),
                "ssa": 0.9 * np.ones(nw) * ureg.dimensionless,
                "phase": phase,
                "attrs": {},
            }

        @pytest.fixture(scope="class")
        def mu_desc(self):
            return np.linspace(1, -1, 11)

        @pytest.fixture(scope="class")
        def mu_asc(self):
            return np.linspace(-1, 1, 11)

        def test_no_check(self, mu_desc):
            """check=None (default) does not raise even if mu is descending."""
            # Following must not raise
            make_aer_core_v2(**self._build_args(mu_desc), check=None)
            make_aer_core_v2(**self._build_args(mu_desc), check="none")

        def test_full(self, mu_desc, mu_asc):
            """check='full' raises iff ValueError mu is strictly descending."""
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**self._build_args(mu_desc), check="full")

            make_aer_core_v2(**self._build_args(mu_asc), check="full")  # must not raise

        def test_fast(self, mu_desc, mu_asc):
            """check='fast' also catches an ill-sorted grid (all rows identical here)."""
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**self._build_args(mu_desc), check="fast")
            make_aer_core_v2(**self._build_args(mu_asc), check="fast")  # must not raise

    class TestNangles:
        """Tests for the nangles parameter in make_aer_core_v2()."""

        def _build_padded_args(self, nangles_vals: np.ndarray):
            """
            Return keyword args for make_aer_core_v2 with NaN-padded arrays.

            The first ``nangles_vals[iw]`` entries of each row are filled with
            ascending mu values; the rest are NaN.
            """
            nw = len(nangles_vals)
            nangle_max = 10
            mu_2d = np.full((nw, nangle_max), np.nan)
            theta_2d = np.full((nw, nangle_max), np.nan)
            phase = np.full((1, nw, nangle_max), np.nan)

            for iw, n in enumerate(nangles_vals):
                mu_row = np.linspace(-1.0, 1.0, n)
                mu_2d[iw, :n] = mu_row
                theta_2d[iw, :n] = np.degrees(np.arccos(mu_row))
                phase[0, iw, :n] = 1.0 / (4.0 * np.pi)

            return {
                "w": np.array([400.0, 550.0, 700.0]) * ureg.nm,
                "phamat": ["11"],
                "mu": mu_2d * ureg.dimensionless,
                "theta": theta_2d * ureg.deg,
                "ext": np.ones(nw) * ureg("1/km"),
                "ssa": 0.9 * np.ones(nw) * ureg.dimensionless,
                "phase": phase * ureg("1/sr"),
                "nangles": nangles_vals,
                "attrs": {},
            }

        def test_nangles_written_to_dataset(self):
            """nangles variable is present and matches input."""
            nangles_vals = np.array([3, 7, 10], dtype=np.int32)
            result = make_aer_core_v2(**self._build_padded_args(nangles_vals))
            assert "nangles" in result
            np.testing.assert_array_equal(result["nangles"].values, nangles_vals)

        def test_check_full_passes_with_nan_tails(self):
            """check='full' validates only the valid portion; NaN tails must not cause failure."""
            nangles_vals = np.array([3, 7, 10], dtype=np.int32)
            make_aer_core_v2(**self._build_padded_args(nangles_vals), check="full")

        def test_check_raises_on_unsorted_valid_portion(self):
            """check='full' raises if the valid portion of any row is not ascending."""
            nangles_vals = np.array([4, 4, 4], dtype=np.int32)
            args = self._build_padded_args(nangles_vals)
            # Reverse the valid portion of row 1 so mu is descending
            mu_arr = args["mu"].m.copy()
            mu_arr[1, :4] = mu_arr[1, :4][::-1]
            args["mu"] = mu_arr * ureg.dimensionless
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**args, check="full")

        def test_none_nangles_not_written(self):
            """When nangles is None (default), no nangles variable is written."""
            nw, nangle = 2, 5
            mu = np.broadcast_to(np.linspace(-1, 1, nangle), (nw, nangle)).copy()
            theta = np.degrees(np.arccos(mu))
            result = make_aer_core_v2(
                w=np.array([400.0, 700.0]) * ureg.nm,
                phamat=["11"],
                mu=mu * ureg.dimensionless,
                theta=theta * ureg.deg,
                ext=np.ones(nw) * ureg("1/km"),
                ssa=0.9 * np.ones(nw) * ureg.dimensionless,
                phase=np.ones((1, nw, nangle)) * ureg("1/sr"),
            )
            assert "nangles" not in result


class TestAerV1ToAerCoreV2:
    class TestSorting:
        """Tests for mu sort order in aer_v1_to_aer_core_v2()."""

        @pytest.fixture
        def ds_descending(self):
            """Aer v1 dataset with mu in descending order (1 → -1, physics convention)."""
            return _make_aer_v1(np.linspace(1.0, -1.0, 37))

        @pytest.fixture
        def ds_ascending(self):
            """Aer v1 dataset with mu already in ascending order (-1 → 1)."""
            return _make_aer_v1(np.linspace(-1.0, 1.0, 37))

        @pytest.mark.parametrize("interp_space", ["mu", "theta"])
        def test_output_always_ascending_mu(self, ds_descending, interp_space):
            """Output mu is strictly ascending regardless of interp_space."""
            result = aer_v1_to_aer_core_v2(ds_descending, interp_space=interp_space)
            mu = result["mu"].values  # (nw, nangle)
            assert np.all(np.diff(mu[0]) > 0), "mu must be strictly ascending"

        def test_already_ascending_input_unchanged(self, ds_ascending):
            """If input mu is already ascending, output should also be ascending."""
            result = aer_v1_to_aer_core_v2(ds_ascending)
            mu = result["mu"].values
            assert np.all(np.diff(mu[0]) > 0)

        def test_phase_reordered_consistently_with_mu(self, ds_descending):
            """Phase values move with the mu grid: check endpoint phase values."""
            ds = ds_descending
            # All phase values are 1/(4π), so the test checks shape/consistency
            result = aer_v1_to_aer_core_v2(ds)
            mu = result["mu"].values[0]  # (nangle,)
            phase = result["phase"].values[0, 0, :]  # (nangle,) for iw=0, phamat=0

            # Identify where mu is closest to -1 and +1
            idx_back = np.argmin(np.abs(mu - (-1.0)))
            idx_fwd = np.argmin(np.abs(mu - 1.0))

            # The input phase is isotropic so all values equal 1/(4π)
            expected = 1.0 / (4.0 * np.pi)
            assert phase[idx_back] == pytest.approx(expected, rel=1e-6)
            assert phase[idx_fwd] == pytest.approx(expected, rel=1e-6)

    class TestIntegrationReal:
        """Integration tests against the real govaerts_2021-desert Aer v1 file."""

        @pytest.fixture(scope="class")
        def ds_input(self):
            from eradiate.data import fresolver

            return fresolver.load_dataset(
                "tests/aerosol/govaerts_2021-desert-aer_v1.nc"
            )

        @pytest.fixture(scope="class")
        def ds_result(self, ds_input):
            return aer_v1_to_aer_core_v2(ds_input)

        def test_mu_ascending(self, ds_result):
            """Converted dataset has mu strictly ascending for all wavelengths."""
            mu = ds_result["mu"].values  # (nw, nangle)
            for iw in range(mu.shape[0]):
                assert np.all(np.diff(mu[iw]) > 0), (
                    f"mu not ascending at wavelength index {iw}"
                )

        def test_output_dimensions(self, ds_input, ds_result):
            """Output has the expected dimensions and sizes."""
            assert "w" in ds_result.dims
            assert "iangle" in ds_result.dims
            assert ds_result.sizes["w"] == ds_input.sizes["w"]
            assert ds_result.sizes["iangle"] == ds_input.sizes["mu"]

        def test_check_full_passes(self, ds_result):
            """Rebuilt dataset passes the check='full' validation."""
            # Re-exercise make_aer_core_v2 with the already-sorted data
            mu = ds_result["mu"].values * ureg.dimensionless
            theta = ds_result["theta"].values * ureg.deg
            ext = ds_result["ext"].values * ureg(ds_result["ext"].attrs["units"])
            ssa = ds_result["ssa"].values * ureg.dimensionless
            phase = ds_result["phase"].values * ureg("1/sr")

            make_aer_core_v2(
                w=ds_result["w"].values * ureg(ds_result["w"].attrs["units"]),
                phamat=list(ds_result["phamat"].values),
                mu=mu,
                theta=theta,
                ext=ext,
                ssa=ssa,
                phase=phase,
                attrs=dict(ds_result.attrs),
                check="full",
            )


class TestLibradtranToAerCoreV2:
    """Tests for libradtran_to_aer_core_v2()."""

    @pytest.fixture(scope="class")
    def ntheta_per_w(self):
        return [3, 5, 4]

    @pytest.fixture(scope="class")
    def ds_input(self, ntheta_per_w):
        return _make_libradtran_ds(ntheta_per_w)

    @pytest.fixture(scope="class")
    def ds_result(self, ds_input):
        return libradtran_to_aer_core_v2(ds_input, check="full")

    def test_nangles_present(self, ds_result):
        """Converted dataset has a nangles variable."""
        assert "nangles" in ds_result

    def test_nangles_values_match_ntheta(self, ds_result, ntheta_per_w):
        """nangles values equal the source ntheta counts."""
        np.testing.assert_array_equal(
            ds_result["nangles"].values, np.array(ntheta_per_w, dtype=np.int32)
        )

    def test_nangles_le_iangle(self, ds_result):
        """Every nangles value is at most iangle_size."""
        assert np.all(ds_result["nangles"].values <= ds_result.sizes["iangle"])

    def test_theta_nan_beyond_nangles(self, ds_result):
        """Theta entries beyond nangles[iw] are NaN."""
        theta = ds_result["theta"].values  # (nw, nthetamax)
        nangles = ds_result["nangles"].values
        for iw, n in enumerate(nangles):
            assert np.all(np.isnan(theta[iw, n:]))

    def test_mu_ascending_in_valid_range(self, ds_result):
        """mu is strictly ascending within the valid range for every wavelength."""
        mu = ds_result["mu"].values  # (nw, nthetamax)
        nangles = ds_result["nangles"].values
        for iw, n in enumerate(nangles):
            assert np.all(np.diff(mu[iw, :n]) > 0), (
                f"mu not ascending at wavelength index {iw}"
            )

    def test_iangle_equals_nthetamax(self, ds_input, ds_result):
        """iangle dimension size matches the source nthetamax."""
        assert ds_result.sizes["iangle"] == ds_input.sizes["nthetamax"]
