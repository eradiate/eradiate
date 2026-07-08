"""Unit tests for eradiate.data.convert."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.data.convert import (
    aer_v1_to_aer_core_v2,
    libradtran_to_aer_core_v2,
    make_aer_core_v2,
)

# ------------------------------------------------------------------------------
#                             Shared dataset builders
# ------------------------------------------------------------------------------


def _isotropic_args(
    nw: int = 2,
    nangle: int = 37,
    scale: float = 1.0,
    mu_1d: np.ndarray | None = None,
) -> dict:
    """
    Keyword args for make_aer_core_v2 with a uniform isotropic phase (p = scale).

    For scale=1 the function is 4π-normalized (∫p dμ = 2). Uses a uniform
    ascending mu grid unless ``mu_1d`` is supplied.
    """
    if mu_1d is None:
        mu_1d = np.linspace(-1.0, 1.0, nangle)
    else:
        nangle = len(mu_1d)
    mu_2d = np.broadcast_to(mu_1d, (nw, nangle)).copy()
    theta_2d = np.degrees(np.arccos(mu_2d))
    return {
        "w": np.linspace(400.0, 700.0, nw) * ureg.nm,
        "phamat": ["11"],
        "mu": mu_2d * ureg.dimensionless,
        "theta": theta_2d * ureg.deg,
        "ext": np.ones(nw) * ureg("1/km"),
        "ssa": 0.9 * np.ones(nw) * ureg.dimensionless,
        "phase": np.full((1, nw, nangle), scale) * ureg("1/sr"),
    }


def _padded_args(nangles_vals: np.ndarray, nangle_max: int = 10) -> dict:
    """
    Keyword args for make_aer_core_v2 with a NaN-padded isotropic phase (p = 1).

    The first ``nangles_vals[iw]`` entries per row are valid; the rest are NaN.
    """
    nw = len(nangles_vals)
    mu_2d = np.full((nw, nangle_max), np.nan)
    theta_2d = np.full((nw, nangle_max), np.nan)
    phase = np.full((1, nw, nangle_max), np.nan)

    for iw, n in enumerate(nangles_vals):
        mu_row = np.linspace(-1.0, 1.0, n)
        mu_2d[iw, :n] = mu_row
        theta_2d[iw, :n] = np.degrees(np.arccos(mu_row))
        phase[0, iw, :n] = 1.0

    return {
        "w": np.linspace(400.0, 700.0, nw) * ureg.nm,
        "phamat": ["11"],
        "mu": mu_2d * ureg.dimensionless,
        "theta": theta_2d * ureg.deg,
        "ext": np.ones(nw) * ureg("1/km"),
        "ssa": 0.9 * np.ones(nw) * ureg.dimensionless,
        "phase": phase * ureg("1/sr"),
        "nangles": nangles_vals,
    }


def _make_aer_v1(
    mu_values: np.ndarray, dtype: np.dtype | type = np.float64
) -> xr.Dataset:
    """
    Build a minimal synthetic Aer v1 dataset with the given mu grid.

    Phase values are isotropic (p = 1, ∫p dμ = 2). Only the (i=0, j=0)
    component is included. Floating-point variables use ``dtype`` so the
    dtype-preservation behaviour of the converter can be exercised.
    """
    nw = 3
    nmu = len(mu_values)
    return xr.Dataset(
        data_vars={
            "sigma_t": ("w", np.ones(nw, dtype=dtype), {"units": "1/km"}),
            "albedo": ("w", (0.9 * np.ones(nw)).astype(dtype), {"units": ""}),
            "phase": (
                ("w", "mu", "i", "j"),
                np.ones((nw, nmu, 1, 1), dtype=dtype),
                {"units": "sr^-1"},
            ),
        },
        coords={
            "w": ("w", np.array([400.0, 550.0, 700.0], dtype=dtype), {"units": "nm"}),
            "mu": ("mu", mu_values.astype(dtype), {"units": ""}),
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

    theta_arr = np.full((nw, nphamat, nthetamax), np.nan)
    phase_arr = np.full((nw, nphamat, nthetamax), np.nan)
    for iw, n in enumerate(ntheta_per_w):
        angles = np.linspace(0.0, 180.0, n)
        theta_arr[iw, 0, :n] = angles
        phase_arr[iw, 0, :n] = 1.0 / (4.0 * np.pi)

    pmom_arr = np.full((nw, nphamat, nmommax), np.nan)
    pmom_arr[:, 0, 0] = 1.0

    return xr.Dataset(
        data_vars={
            "wavelen": ("nlam", np.linspace(400.0, 700.0, nw), {"units": "nm"}),
            "ext": ("nlam", np.ones(nw), {"units": "1/km"}),
            "ssa": ("nlam", 0.9 * np.ones(nw), {"units": ""}),
            "ntheta": (
                ("nlam", "nphamat"),
                np.array([[n] for n in ntheta_per_w], dtype=np.int32),
            ),
            "theta": (("nlam", "nphamat", "nthetamax"), theta_arr),
            "phase": (("nlam", "nphamat", "nthetamax"), phase_arr),
            "pmom": (("nlam", "nphamat", "nmommax"), pmom_arr),
        }
    )


# ------------------------------------------------------------------------------
#                                      Tests
# ------------------------------------------------------------------------------


class TestMakeAerCoreV2:
    class TestCheck:
        """Tests for the mu sort-order check in make_aer_core_v2()."""

        @pytest.fixture(scope="class")
        @classmethod
        def mu_desc(cls):
            return np.linspace(1, -1, 11)

        @pytest.fixture(scope="class")
        @classmethod
        def mu_asc(cls):
            return np.linspace(-1, 1, 11)

        def test_no_check(self, mu_desc):
            """check=None and check='none' do not raise even for descending mu."""
            make_aer_core_v2(**_isotropic_args(mu_1d=mu_desc), check=None)
            make_aer_core_v2(**_isotropic_args(mu_1d=mu_desc), check="none")

        def test_full_raises_on_descending(self, mu_desc):
            """check='full' raises ValueError for a descending mu grid."""
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**_isotropic_args(mu_1d=mu_desc), check="full")

        def test_full_passes_on_ascending(self, mu_asc):
            """check='full' does not raise for a valid ascending mu grid."""
            make_aer_core_v2(**_isotropic_args(mu_1d=mu_asc), check="full")

        def test_fast_raises_on_descending(self, mu_desc):
            """check='fast' catches an ill-sorted grid."""
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**_isotropic_args(mu_1d=mu_desc), check="fast")

    class TestNangles:
        """Tests for the nangles parameter in make_aer_core_v2()."""

        @pytest.fixture(scope="class")
        @classmethod
        def nangles_vals(cls):
            return np.array([3, 7, 10], dtype=np.int32)

        def test_nangles_written(self, nangles_vals):
            """nangles variable is present and matches input."""
            ds = make_aer_core_v2(**_padded_args(nangles_vals))
            np.testing.assert_array_equal(ds["nangles"].values, nangles_vals)

        def test_check_full_ignores_nan_tails(self, nangles_vals):
            """check='full' validates only the valid portion; NaN tails do not cause failure."""
            make_aer_core_v2(**_padded_args(nangles_vals), check="full")

        def test_check_raises_on_unsorted_valid_portion(self, nangles_vals):
            """check='full' raises if the valid portion of any row is not ascending."""
            args = _padded_args(nangles_vals)
            mu_arr = args["mu"].m.copy()
            n = nangles_vals[1]
            mu_arr[1, :n] = mu_arr[1, :n][::-1]
            args["mu"] = mu_arr * ureg.dimensionless
            with pytest.raises(ValueError, match="strictly ascending"):
                make_aer_core_v2(**args, check="full")

        def test_nangles_inferred_from_phase(self):
            """When nangles is None it is inferred by counting non-NaN entries in phase."""
            nw, nangle = 2, 5
            ds = make_aer_core_v2(**_isotropic_args(nw=nw, nangle=nangle))
            np.testing.assert_array_equal(ds["nangles"].values, np.full(nw, nangle))

    class TestAutoPmom:
        """Tests for automatic Legendre coefficient computation in make_aer_core_v2()."""

        @pytest.fixture(scope="class")
        @classmethod
        def base_args(cls):
            return _isotropic_args(nw=3, nangle=37)

        def test_default_129_moments(self, base_args):
            """pmom is computed with 129 moments by default."""
            nw = base_args["w"].shape[0]
            ds = make_aer_core_v2(**base_args)
            assert ds["pmom"].shape == (nw, 129)
            np.testing.assert_array_equal(ds["nmom"].values, np.full(nw, 129))

        def test_scalar_nmom(self, base_args):
            """pmom has the requested moment count when nmom is an int."""
            nw = base_args["w"].shape[0]
            ds = make_aer_core_v2(**base_args, nmom=64)
            assert ds["pmom"].shape == (nw, 64)
            np.testing.assert_array_equal(ds["nmom"].values, np.full(nw, 64))

        def test_explicit_pmom_stored_verbatim(self, base_args):
            """Explicitly provided pmom is stored as-is; nmom is inferred from non-NaN count."""
            nw = base_args["w"].shape[0]
            sentinel = np.full((nw, 10), 42.0)
            ds = make_aer_core_v2(**base_args, pmom=sentinel)
            np.testing.assert_array_equal(ds["pmom"].values, sentinel)
            np.testing.assert_array_equal(ds["nmom"].values, np.full(nw, 10))

        def test_explicit_pmom_nan_padded_nmom_inferred(self, base_args):
            """nmom is inferred per-wavelength for NaN-padded explicit pmom."""
            nw = base_args["w"].shape[0]
            nmom_per_w = np.array([4, 7, 10], dtype=np.int32)[:nw]
            pmom_arr = np.full((nw, int(nmom_per_w.max())), np.nan)
            for iw, n in enumerate(nmom_per_w):
                pmom_arr[iw, :n] = 1.0
            ds = make_aer_core_v2(**base_args, pmom=pmom_arr)
            np.testing.assert_array_equal(ds["nmom"].values, nmom_per_w)

        def test_nmom_ndarray_without_pmom_raises(self, base_args):
            """ValueError when nmom is an ndarray but pmom is not provided."""
            nw = base_args["w"].shape[0]
            with pytest.raises(
                ValueError,
                match="Pass an ndarray only together with an explicit 'pmom'",
            ):
                make_aer_core_v2(**base_args, nmom=np.full(nw, 10, dtype=np.int32))

        def test_isotropic_pmom_values(self, base_args):
            """For isotropic p=1, l=0 coefficient is 1 and l>0 are ~0."""
            ds = make_aer_core_v2(**base_args, nmom=20)
            pmom_vals = ds["pmom"].values[0, :]  # first wavelength
            np.testing.assert_allclose(pmom_vals[0], 1.0, rtol=1e-6)
            np.testing.assert_allclose(pmom_vals[1:], 0.0, atol=1e-6)

        def test_nan_padded_phase(self):
            """Auto-computation handles NaN-padded angular arrays correctly."""
            nangles_vals = np.array([5, 10], dtype=np.int32)
            ds = make_aer_core_v2(**_padded_args(nangles_vals, nangle_max=10), nmom=10)
            assert ds["pmom"].shape == (len(nangles_vals), 10)
            np.testing.assert_allclose(ds["pmom"].values[:, 0], 1.0, rtol=1e-6)

    class TestNormalize:
        """Tests for the normalize parameter in make_aer_core_v2()."""

        def test_false_leaves_phase_unchanged(self):
            """normalize=False (default) stores phase as-is."""
            nw = 2
            args = _isotropic_args(nw=nw, scale=4.0)
            ds = make_aer_core_v2(**args, normalize=False, pmom=np.zeros((nw, 2)))
            np.testing.assert_allclose(ds["phase"].values, 4.0, rtol=1e-12)

        def test_true_enforces_integral(self):
            """normalize=True rescales p11 so that ∫p11 dμ = 2."""
            args = _isotropic_args(nw=2, scale=4.0)
            ds = make_aer_core_v2(**args, normalize=True)
            mu = ds["mu"].values
            phase = ds["phase"].values
            for iw in range(mu.shape[0]):
                integral = np.trapezoid(phase[0, iw, :], mu[iw, :])
                np.testing.assert_allclose(integral, 2.0, rtol=1e-12)

        def test_scales_all_components(self):
            """All phase matrix components are rescaled by the same factor."""
            nw, nangle = 2, 37
            mu_1d = np.linspace(-1.0, 1.0, nangle)
            mu_2d = np.broadcast_to(mu_1d, (nw, nangle)).copy()
            # p11=4, p12=2 → after normalization (factor 0.5) → p11=1, p12=0.5
            phase = np.stack(
                [np.full((nw, nangle), 4.0), np.full((nw, nangle), 2.0)]
            ) * ureg("1/sr")
            ds = make_aer_core_v2(
                w=np.linspace(400.0, 700.0, nw) * ureg.nm,
                phamat=["11", "12"],
                mu=mu_2d * ureg.dimensionless,
                theta=np.degrees(np.arccos(mu_2d)) * ureg.deg,
                ext=np.ones(nw) * ureg("1/km"),
                ssa=0.9 * np.ones(nw) * ureg.dimensionless,
                phase=phase,
                normalize=True,
                pmom=np.zeros((nw, 2)),
            )
            ratio = (
                ds["phase"].sel(phamat="12").values
                / ds["phase"].sel(phamat="11").values
            )
            np.testing.assert_allclose(ratio, 0.5, rtol=1e-12)

        def test_zero_integral_raises(self):
            """normalize=True raises ValueError when p11 integrates to 0."""
            args = _isotropic_args(nw=1, scale=0.0)
            with pytest.raises(ValueError, match="integrates to 0"):
                make_aer_core_v2(**args, normalize=True)


class TestAerV1ToAerCoreV2:
    class TestSorting:
        """Tests for mu sort order in aer_v1_to_aer_core_v2()."""

        @pytest.fixture(scope="class")
        @classmethod
        def ds_descending(cls):
            return _make_aer_v1(np.linspace(1.0, -1.0, 37))

        @pytest.fixture(scope="class")
        @classmethod
        def ds_ascending(cls):
            return _make_aer_v1(np.linspace(-1.0, 1.0, 37))

        def test_output_mu_ascending(self, ds_descending, ds_ascending):
            """Output mu is always strictly ascending regardless of input order."""
            for ds in [ds_descending, ds_ascending]:
                result = aer_v1_to_aer_core_v2(ds)
                assert np.all(np.diff(result["mu"].values[0]) > 0)

        def test_phase_consistent_with_mu(self, ds_descending):
            """Phase values are reordered consistently with the mu grid."""
            result = aer_v1_to_aer_core_v2(ds_descending)
            phase = result["phase"].values[0, 0, :]  # phamat=0, iw=0
            np.testing.assert_allclose(phase, 1.0, rtol=1e-12)

    class TestDtype:
        """Tests for the dtype parameter in aer_v1_to_aer_core_v2()."""

        @pytest.mark.parametrize(
            "source_dtype, dtype_arg, expected",
            [
                # dtype=None preserves the origin dtype (no silent upcast)
                (np.float32, None, np.float32),
                (np.float64, None, np.float64),
                # explicit dtype casts regardless of the origin dtype
                (np.float64, np.float32, np.float32),
                (np.float32, np.float64, np.float64),
            ],
        )
        def test_dtype_handling(self, source_dtype, dtype_arg, expected):
            ds = _make_aer_v1(np.linspace(-1.0, 1.0, 37), dtype=source_dtype)
            result = aer_v1_to_aer_core_v2(ds, dtype=dtype_arg)
            for var in ("ext", "ssa", "phase"):
                assert result[var].dtype == expected, (
                    f"{var}: expected {expected}, got {result[var].dtype}"
                )

    class TestIntegrationReal:
        """Integration tests against the real govaerts_2021-desert Aer v1 file."""

        @pytest.fixture(scope="class")
        @classmethod
        def ds_input(cls):
            from eradiate.data import fresolver

            return fresolver.load_dataset(
                "tests/aerosol/govaerts_2021-desert-aer_v1.nc"
            )

        @pytest.fixture(scope="class")
        @classmethod
        def ds_result(cls, ds_input):
            return aer_v1_to_aer_core_v2(ds_input)

        def test_mu_ascending(self, ds_result):
            """Converted dataset has mu strictly ascending for all wavelengths."""
            mu = ds_result["mu"].values
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


class TestLibradtranToAerCoreV2:
    """Tests for libradtran_to_aer_core_v2()."""

    @pytest.fixture(scope="class")
    @classmethod
    def ntheta_per_w(cls):
        return [3, 5, 4]

    @pytest.fixture(scope="class")
    @classmethod
    def ds_input(cls, ntheta_per_w):
        return _make_libradtran_ds(ntheta_per_w)

    @pytest.fixture(scope="class")
    @classmethod
    def ds_result(cls, ds_input):
        return libradtran_to_aer_core_v2(ds_input, check="full")

    def test_nangles_match_ntheta(self, ds_result, ntheta_per_w):
        """nangles values match the source ntheta counts."""
        np.testing.assert_array_equal(
            ds_result["nangles"].values, np.array(ntheta_per_w, dtype=np.int32)
        )

    def test_theta_nan_beyond_nangles(self, ds_result):
        """Theta entries beyond nangles[iw] are NaN."""
        theta = ds_result["theta"].values
        for iw, n in enumerate(ds_result["nangles"].values):
            assert np.all(np.isnan(theta[iw, n:]))

    def test_mu_ascending_in_valid_range(self, ds_result):
        """mu is strictly ascending within the valid range for every wavelength."""
        mu = ds_result["mu"].values
        for iw, n in enumerate(ds_result["nangles"].values):
            assert np.all(np.diff(mu[iw, :n]) > 0), (
                f"mu not ascending at wavelength index {iw}"
            )

    def test_iangle_equals_nthetamax(self, ds_input, ds_result):
        """iangle dimension size matches the source nthetamax (single component)."""
        assert ds_result.sizes["iangle"] == ds_input.sizes["nthetamax"]

    class TestUnionGrid:
        """Tests that the union angular grid across phase-matrix components is used."""

        @pytest.fixture(scope="class")
        @classmethod
        def ds_input(cls):
            # Component 0: [0, 90, 180]°; Component 1: [45, 135]° — fully disjoint
            nw, nphamat, nthetamax = 2, 2, 3
            theta_arr = np.full((nw, nphamat, nthetamax), np.nan)
            phase_arr = np.full((nw, nphamat, nthetamax), np.nan)
            ntheta_arr = np.zeros((nw, nphamat), dtype=np.int32)
            theta_arr[:, 0, :3] = [0.0, 90.0, 180.0]
            phase_arr[:, 0, :3] = 1.0 / (4.0 * np.pi)
            ntheta_arr[:, 0] = 3
            theta_arr[:, 1, :2] = [45.0, 135.0]
            phase_arr[:, 1, :2] = 1.0 / (4.0 * np.pi)
            ntheta_arr[:, 1] = 2
            pmom_arr = np.full((nw, nphamat, 4), np.nan)
            pmom_arr[:, 0, 0] = 1.0
            return xr.Dataset(
                data_vars={
                    "wavelen": ("nlam", np.linspace(400.0, 700.0, nw), {"units": "nm"}),
                    "ext": ("nlam", np.ones(nw), {"units": "1/km"}),
                    "ssa": ("nlam", 0.9 * np.ones(nw), {"units": ""}),
                    "ntheta": (("nlam", "nphamat"), ntheta_arr),
                    "theta": (("nlam", "nphamat", "nthetamax"), theta_arr),
                    "phase": (("nlam", "nphamat", "nthetamax"), phase_arr),
                    "pmom": (("nlam", "nphamat", "nmommax"), pmom_arr),
                }
            )

        @pytest.fixture(scope="class")
        @classmethod
        def ds_result(cls, ds_input):
            return libradtran_to_aer_core_v2(ds_input, check="full")

        def test_nangles_is_union_size(self, ds_result):
            """nangles equals the count of distinct mu values from all components."""
            # union of [0, 90, 180]° and [45, 135]° → 5 distinct angles
            np.testing.assert_array_equal(ds_result["nangles"].values, [5, 5])
