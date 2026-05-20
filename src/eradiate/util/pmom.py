"""
Legendre moment decomposition of a phase function.

Python translation of pmom.c from libRadtran (Claudia Emde, DLR).
Implements the exact Buras-Dowling-Emde (BDE) method and a trapezoidal fallback.

Reference
---------
Buras, R., Dowling, T., & Emde, C. (2011). New secondary-scattering correction
in DISORT with increased efficiency for forward scattering. Journal of
Quantitative Spectroscopy and Radiative Transfer, 112(12), 2028–2034.
"""

from __future__ import annotations

import warnings

import click
import numpy as np


def calc_pmom_bde(mu: np.ndarray, phase: np.ndarray, nmom: int) -> np.ndarray:
    """Compute Legendre moments using the BDE exact method.

    Exact for piecewise-linear phase functions. Uses the jump-condition
    (telescoping sum) approach from Buras, Dowling & Emde (2011): for each
    interval a linear fit C+D*mu is computed, and the resulting boundary
    terms are accumulated analytically using Legendre recurrence.

    Parameters
    ----------
    mu : ndarray, shape (ntheta,)
        Cosine of scattering angle, ordered from +1 (0°) to −1 (180°).
    phase : ndarray, shape (ntheta,)
        Phase function values at ``mu``.
    nmom : int
        Number of moments to compute (l = 0 … nmom−1).

    Returns
    -------
    ndarray, shape (nmom,)
        Raw moment array. Apply a factor of −0.5 to obtain ξ_l.

    Notes
    -----
    The factor −0.5 corrects for the decreasing-mu convention (integrals come
    out with the wrong sign) and the half-interval normalization of Legendre
    moments. It is applied by :func:`compute_pmom`.
    """
    ntheta = len(mu)
    mu2 = mu * mu
    pmom = np.zeros(nmom)

    # Zeroth moment: trapezoidal (mirrors the separate loop in calc_pmom)
    pmom[0] = 0.5 * np.dot(phase[1:] + phase[:-1], np.diff(mu))

    if nmom < 2:
        return pmom

    # Piecewise-linear fit per interval: phase(mu) ≈ C_i + D_i * mu
    dmu = np.diff(mu)  # negative (mu decreasing)
    D_iv = np.diff(phase) / dmu  # shape (ntheta-1,)
    C_iv = phase[:-1] - mu[:-1] * D_iv  # shape (ntheta-1,)

    # Jump at each grid point: dC[i] = C_{i-1} − C_i
    # Boundary: C = 0 outside the grid (no interval to left/right)
    dC = np.empty(ntheta)
    dC[0] = -C_iv[0]
    dC[1:-1] = C_iv[:-1] - C_iv[1:]
    dC[-1] = C_iv[-1]

    dD = np.empty(ntheta)
    dD[0] = -D_iv[0]
    dD[1:-1] = D_iv[:-1] - D_iv[1:]
    dD[-1] = D_iv[-1]

    # First moment (special closed-form, from paper)
    pmom[1] = np.dot(0.5 * dC * mu2 + (1.0 / 3.0) * dD * mu2 * mu, np.ones(ntheta))

    if nmom < 3:
        return pmom

    # Higher moments l = 2 … nmom-1
    # Rolling Legendre recurrence: track P_l, P_{l+1}, P_{l+2} as shape-(ntheta,) arrays.
    # At each iteration l we need P_l, P_{l+1}, P_{l+2}; advance by 1 after use.
    pa = np.ones(ntheta)  # P_0
    pb = mu.copy()  # P_1
    pc = (3.0 * mu2 - 1.0) / 2.0  # P_2
    pd = ((2 * 3 - 1) * mu * pc - 2 * pb) / 3  # P_3
    pe = ((2 * 4 - 1) * mu * pd - 3 * pc) / 4  # P_4

    pl, pl1, pl2 = pc, pd, pe  # P_l, P_{l+1}, P_{l+2} for l=2

    for l in range(2, nmom):
        pmom[l] = np.dot(
            dC * (pl1 - mu * pl) / l
            + dD * ((pl1 * mu * (l + 2) - pl2) / (l + 1) - pl * mu2) / (l - 1),
            np.ones(ntheta),
        )
        pl2_new = ((2 * (l + 3) - 1) * mu * pl2 - (l + 2) * pl1) / (l + 3)
        pl, pl1, pl2 = pl1, pl2, pl2_new

    return pmom


def calc_pmom_trap(mu: np.ndarray, phase: np.ndarray, nmom: int) -> np.ndarray:
    """Compute Legendre moments using trapezoidal quadrature.

    Parameters
    ----------
    mu : ndarray, shape (ntheta,)
        Cosine of scattering angle, ordered from +1 (0°) to −1 (180°).
    phase : ndarray, shape (ntheta,)
        Phase function values at ``mu``.
    nmom : int
        Number of moments to compute (l = 0 … nmom−1).

    Returns
    -------
    ndarray, shape (nmom,)
        Legendre moments ξ_l (normalized; no extra factor needed).
    """
    ntheta = len(mu)
    abs_dmu = np.abs(np.diff(mu))
    pmom = np.zeros(nmom)

    p_prev = np.ones(ntheta)  # P_0
    pmom[0] = 0.25 * np.dot(p_prev[:-1] * phase[:-1] + p_prev[1:] * phase[1:], abs_dmu)

    if nmom < 2:
        return pmom

    p_curr = mu.copy()  # P_1
    pmom[1] = 0.25 * np.dot(p_curr[:-1] * phase[:-1] + p_curr[1:] * phase[1:], abs_dmu)

    for l in range(2, nmom):
        p_next = ((2 * l - 1) * mu * p_curr - (l - 1) * p_prev) / l
        pmom[l] = 0.25 * np.dot(
            p_next[:-1] * phase[:-1] + p_next[1:] * phase[1:], abs_dmu
        )
        p_prev = p_curr
        p_curr = p_next

    return pmom


def compute_pmom(
    theta: np.ndarray,
    phase: np.ndarray,
    nmom: int = 1000,
    grid_res: int = 3,
    normalize: bool = False,
    coefficients: bool = False,
) -> np.ndarray:
    """Compute Legendre moments of a phase function.

    Parameters
    ----------
    theta : ndarray, shape (ntheta,)
        Scattering angles in degrees, 0 to 180.
    phase : ndarray, shape (ntheta,)
        Phase function values.
    nmom : int
        Number of Legendre moments to compute.
    grid_res : int
        Angular grid resolution (1–5). ``3`` (default) uses the input grid
        directly with the exact BDE method. Other values interpolate onto a
        fixed equidistant grid and use trapezoidal integration.

        =====  =============================================
        Value  Grid
        =====  =============================================
        1      equidistant 0.01°
        2      equidistant 0.001°
        3      input grid (BDE method, recommended)
        4      0.001° for 0–10°, 0.01° for 10–180°
        5      0.01° for 0–10°, 0.1° for 10–180° (coarse)
        =====  =============================================

    normalize : bool
        If ``True``, rescale phase function to integrate to 2 over [−1, 1]
        before computing moments.
    coefficients : bool
        If ``True``, multiply each moment by (2l+1) to obtain the Legendre
        expansion coefficients required by e.g. DISORT/uvspec.

    Returns
    -------
    ndarray, shape (nmom,)
        Legendre moments ξ_l, or expansion coefficients (2l+1)ξ_l if
        ``coefficients=True``.
    """
    theta = np.asarray(theta, dtype=float)
    phase = np.asarray(phase, dtype=float)
    mu = np.cos(theta * np.pi / 180.0)

    pint = 0.5 * np.dot(phase[1:] + phase[:-1], np.abs(np.diff(mu)))
    if abs(2.0 - pint) > 1e-2:
        warnings.warn(
            f"Phase function not normalized to 2 but to {pint:.8f}",
            stacklevel=2,
        )

    if normalize:
        phase = phase * (2.0 / pint)

    if grid_res == 3:
        raw = calc_pmom_bde(mu, phase, nmom)
        result = -0.5 * raw
    else:
        if grid_res == 1:
            theta_grid = np.arange(0.0, 180.0 + 1e-9, 0.01)
        elif grid_res == 2:
            theta_grid = np.arange(0.0, 180.0 + 1e-9, 0.001)
        elif grid_res == 4:
            theta_grid = np.concatenate(
                [np.arange(0.0, 10.0, 0.001), np.arange(10.0, 180.0 + 1e-9, 0.01)]
            )
        elif grid_res == 5:
            theta_grid = np.concatenate(
                [np.arange(0.0, 10.0, 0.01), np.arange(10.0, 180.0 + 1e-9, 0.1)]
            )
        else:
            raise ValueError(f"grid_res must be 1–5, got {grid_res}")

        mu_grid = np.cos(theta_grid * np.pi / 180.0)
        # Linear interpolation on the input theta grid (C uses piecewise-linear spline)
        phase_grid = np.interp(theta_grid, theta, phase)
        result = calc_pmom_trap(mu_grid, phase_grid, nmom)

    if coefficients:
        result *= 2 * np.arange(nmom) + 1

    return result


@click.command()
@click.argument("filename", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-l",
    "nmom",
    default=1000,
    show_default=True,
    metavar="N",
    help="Number of Legendre moments.",
)
@click.option(
    "-r",
    "grid_res",
    default=3,
    show_default=True,
    type=click.Choice(["1", "2", "3", "4", "5"]),
    metavar="G",
    help="Angular grid resolution (3 = input grid, BDE method).",
)
@click.option(
    "-c",
    "coefficients",
    is_flag=True,
    help="Output coefficients (2l+1)ξ_l instead of moments, as required by uvspec.",
)
@click.option(
    "-n",
    "normalize",
    is_flag=True,
    help="Normalize phase function to 2 before computing moments.",
)
def main(
    filename: str, nmom: int, grid_res: str, coefficients: bool, normalize: bool
) -> None:
    """Calculate Legendre moments of a phase function.

    FILENAME must be a 2-column text file: scattering angle in degrees (col 1)
    and phase function value (col 2).
    """
    data = np.loadtxt(filename)
    if data.ndim != 2 or data.shape[1] < 2:
        raise click.BadParameter(f"{filename} must be a 2-column file")

    theta, phase = data[:, 0], data[:, 1]
    click.echo(f" ... read {len(theta)} data points from {filename}", err=True)
    click.echo(f"... calculate {nmom} Legendre moments", err=True)
    click.echo(f"... use grid resolution {grid_res}", err=True)

    xi = compute_pmom(
        theta,
        phase,
        nmom=nmom,
        grid_res=int(grid_res),
        normalize=normalize,
        coefficients=coefficients,
    )

    for val in xi:
        click.echo(val)


if __name__ == "__main__":
    main()
