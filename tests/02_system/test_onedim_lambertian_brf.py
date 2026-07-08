"""Test cases with OneDimSolverApp and a Lambertian surface."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eradiate


def test_onedim_lambertian_brf(mode_mono_double, artefact_dir, plot_figures):
    r"""
    Measured lambertian BRF
    =======================

    This test case checks that measured BRF matches the Lambertian surface's
    reflectance.

    Rationale
    ---------

    * Geometry: a square surface with unit size and a Lambertian BRDF with
      reflectance :math:`\rho \in [0.0, 0.3, 0.5, 0.7, 1.0]`.
    * Illumination: Directional illumination with a zenith angle
      :math:`\theta \in [0.0, 30.0, 60.0]°`.
    * Sensor: Multi-distant measure covering a plane (1001 angular points,
      1 sample per pixel) and targeting (0, 0, 0).
      We take an odd number of angular points to ensure that the special value
      :math:`\theta = 0°` is included.


    Expected behaviour
    ------------------

    The BRF results must be equal to reflectance input values.

    Results
    -------

    .. image:: generated/plots/test_onedim_lambertian_brf_0.0.png
       :width: 75%

    .. image:: generated/plots/test_onedim_lambertian_brf_30.0.png
       :width: 75%

    .. image:: generated/plots/test_onedim_lambertian_brf_60.0.png
       :width: 75%
    """
    spp = 1
    n_vza = 51
    illumination_zenith_values = [0.0, 30.0, 60.0]
    reflectance_values = [1.0, 0.7, 0.5, 0.3, 0.0]

    results = {}
    for illumination_zenith in illumination_zenith_values:
        results[illumination_zenith] = {}
        for reflectance in reflectance_values:
            # Run simulation
            exp = eradiate.experiments.AtmosphereExperiment(
                illumination={
                    "type": "directional",
                    "zenith": illumination_zenith,
                    "azimuth": 0.0,
                },
                measures={
                    "type": "mdistant",
                    "id": "toa_pplane",
                    "construct": "hplane",
                    "zeniths": np.linspace(-90, 90, n_vza),
                    "azimuth": 0.0,
                    "spp": spp,
                },
                surface={
                    "type": "lambertian",
                    "reflectance": reflectance,
                },
                atmosphere=None,
            )
            eradiate.run(exp)

            results[illumination_zenith][reflectance] = exp.results["toa_pplane"]

    # Plot result
    if plot_figures:
        for illumination_zenith in illumination_zenith_values:
            fig, ax1 = plt.subplots(figsize=(6, 3), layout="constrained")

            with plt.rc_context({"lines.linestyle": ":", "lines.marker": "."}):
                for reflectance in reflectance_values:
                    results[illumination_zenith][reflectance].brf.plot(ax=ax1, x="vza")

            ax1.set_xlabel("Signed viewing zenith angle [°]")
            ax1.set_xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
            ax1.set_ylabel("BRF [dimensionless]")
            ax1.set_title(rf"$\theta$ = {illumination_zenith}°")
            ax1.legend(
                [f"{reflectance}" for reflectance in reflectance_values],
                title=r"$\rho$",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

            outdir = Path(artefact_dir) / "plots"
            outdir.mkdir(parents=True, exist_ok=True)
            filename = f"test_onedim_lambertian_brf_{illumination_zenith}.png"
            fname_plot = outdir / filename
            fig.savefig(fname_plot, dpi=200, bbox_inches="tight")

            plt.close()

    for illumination_zenith in illumination_zenith_values:
        for reflectance in reflectance_values:
            assert np.allclose(
                results[illumination_zenith][reflectance].brf, reflectance
            )
