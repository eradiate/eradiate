"""Tests for eradiate.util.pmom, verified against the libRadtran C reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from eradiate import fresolver
from eradiate.util.pmom import compute_pmom

REFERENCE_DIR = fresolver.resolve("tests/pmom_references")

# Absolute tolerance for comparison against C reference outputs.
# Differences arise from floating-point ordering; the algorithms are identical.
ATOL_REFERENCE = 1e-5


# ------------------------------------------------------------------------------
#                                     Helpers
# ------------------------------------------------------------------------------


def _load_phase(case: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(REFERENCE_DIR / f"{case}_phase.dat")
    return data[:, 0], data[:, 1]


def _load_moments(case: str, grid_res: int) -> np.ndarray:
    return np.loadtxt(REFERENCE_DIR / f"{case}_pmom_r{grid_res}.dat")


def _load_coefficients(case: str) -> np.ndarray:
    # C outputs all coefficients on one line separated by spaces
    text = (REFERENCE_DIR / f"{case}_coeff_r3.dat").read_text()
    return np.array(text.split(), dtype=float)


class TestPMom:
    # --------------------------------------------------------------------------
    #                                Fixtures
    # --------------------------------------------------------------------------

    @pytest.fixture
    def rayleigh_phase(self) -> tuple[np.ndarray, np.ndarray]:
        """Rayleigh p(μ)=¾(1+μ²) on a 1001-point grid, normalized to 2."""
        theta = np.linspace(0, 180, 1001)
        mu = np.cos(np.radians(theta))
        return theta, 0.75 * (1 + mu**2)

    @pytest.fixture
    def theta_mu_fine(self) -> tuple[np.ndarray, np.ndarray]:
        """10001-point theta grid and corresponding mu values."""
        theta = np.linspace(0, 180, 10001)
        return theta, np.cos(np.radians(theta))

    # --------------------------------------------------------------------------
    #         Reference comparisons (C binary output saved in reference/)
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("case", ["rayleigh", "hg05", "hg85"])
    @pytest.mark.parametrize("grid_res", [1, 3])
    def test_moments_vs_reference(self, case: str, grid_res: int) -> None:
        theta, phase = _load_phase(case)
        ref = _load_moments(case, grid_res)
        result = compute_pmom(theta, phase, nmom=64, grid_res=grid_res)
        np.testing.assert_allclose(result, ref, atol=ATOL_REFERENCE, rtol=0)

    @pytest.mark.parametrize("case", ["rayleigh", "hg05", "hg85"])
    def test_coefficients_vs_reference(self, case: str) -> None:
        theta, phase = _load_phase(case)
        ref = _load_coefficients(case)
        result = compute_pmom(theta, phase, nmom=64, grid_res=3, coefficients=True)
        np.testing.assert_allclose(result, ref, atol=ATOL_REFERENCE, rtol=0)

    # --------------------------------------------------------------------------
    #                               Analytic checks
    # --------------------------------------------------------------------------

    def test_rayleigh_analytic_moments(self, theta_mu_fine) -> None:
        """Rayleigh p(μ)=¾(1+μ²) has exact moments: ξ₀=1, ξ₁=0, ξ₂=1/10, rest 0."""
        theta, mu = theta_mu_fine
        xi = compute_pmom(theta, 0.75 * (1 + mu**2), nmom=7, grid_res=3)
        np.testing.assert_allclose(xi[0], 1.0, atol=1e-8)
        np.testing.assert_allclose(xi[1], 0.0, atol=1e-8)
        np.testing.assert_allclose(xi[2], 0.1, atol=1e-8)
        np.testing.assert_allclose(xi[3:], 0.0, atol=1e-8)

    @pytest.mark.parametrize("g", [0.0, 0.5, 0.75])
    def test_henyey_greenstein_moments(self, g: float, theta_mu_fine) -> None:
        """HG(g) has analytic moments ξ_l = g^l. Tests normalization and asymmetry."""
        theta, mu = theta_mu_fine
        phase = (1 - g**2) / (1 + g**2 - 2 * g * mu) ** 1.5
        xi = compute_pmom(theta, phase, nmom=8, grid_res=3)
        np.testing.assert_allclose(xi, g ** np.arange(8), atol=1e-6)

    # --------------------------------------------------------------------------
    #                           Normalization behaviour
    # --------------------------------------------------------------------------

    def test_normalization_warning(self, rayleigh_phase) -> None:
        """A phase function whose integral deviates from 2 by >1% triggers a warning."""
        theta, phase = rayleigh_phase
        with pytest.warns(UserWarning, match="Phase function not normalized"):
            compute_pmom(theta, phase * 2.0, nmom=4, grid_res=3)

    def test_normalize_flag_restores_xi0(self, rayleigh_phase) -> None:
        """normalize=True should rescale so ξ₀=1 regardless of input amplitude."""
        theta, phase = rayleigh_phase
        with pytest.warns(UserWarning, match="Phase function not normalized"):
            xi = compute_pmom(theta, phase * 3.7, nmom=4, grid_res=3, normalize=True)
        np.testing.assert_allclose(xi[0], 1.0, atol=1e-8)

    def test_normalize_preserves_shape(self, rayleigh_phase) -> None:
        """normalize=True should not change the relative shape of the moments."""
        theta, phase = rayleigh_phase
        xi_base = compute_pmom(theta, phase, nmom=6, grid_res=3)
        with pytest.warns(UserWarning):
            xi_scaled = compute_pmom(
                theta, phase * 5.0, nmom=6, grid_res=3, normalize=True
            )
        np.testing.assert_allclose(xi_scaled, xi_base, atol=1e-6)

    # --------------------------------------------------------------------------
    #                                Coefficients
    # --------------------------------------------------------------------------

    def test_coefficients_flag(self, rayleigh_phase) -> None:
        """coefficients=True should multiply moment l by (2l+1)."""
        theta, phase = rayleigh_phase
        xi = compute_pmom(theta, phase, nmom=6, grid_res=3)
        coeff = compute_pmom(theta, phase, nmom=6, grid_res=3, coefficients=True)
        orders = np.arange(6)
        np.testing.assert_allclose(coeff, (2 * orders + 1) * xi, rtol=1e-12)
