"""Test cases with OneDimSolverApp and a lambertian surface."""

import os

import matplotlib.pyplot as plt
import numpy as np

import eradiate

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def test_onedim_lambertian_brf(mode_mono_double):
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

    The BRF results must agree with the reflectance input values within a
    relative tolerance of 0.1%.

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
    n_vza = 1001
    illumination_zenith_values = [0.0, 30.0, 60.0]
    reflectance_values = [1.0, 0.7, 0.5, 0.3, 0.0]

    results = {}
    for illumination_zenith in illumination_zenith_values:
        results[illumination_zenith] = {}
        for reflectance in reflectance_values:
            # Run simulation
            exp = eradiate.experiments.OneDimExperiment(
                illumination={
                    "type": "directional",
                    "zenith": illumination_zenith,
                    "azimuth": 0.0,
                },
                measures={
                    "type": "distant",
                    "id": "toa_pplane",
                    "construct": "from_viewing_angles",
                    "zeniths": np.linspace(-90, 90, n_vza),
                    "azimuths": 0.0,
                    "spp": spp,
                },
                surface={
                    "type": "lambertian",
                    "reflectance": reflectance,
                },
                atmosphere=None,
            )
            exp.run()

            results[illumination_zenith][reflectance] = exp.results["toa_pplane"]

    # Plot results
    for illumination_zenith in illumination_zenith_values:
        fig = plt.figure(figsize=(6, 3))
        ax1 = plt.gca()
        for reflectance in reflectance_values:
            results[illumination_zenith][reflectance].brf.plot(ax=ax1, x="vza")
        filename = f"test_onedim_lambertian_brf_{illumination_zenith}.png"
        ensure_output_dir(os.path.join(output_dir, "plots"))
        fname_plot = os.path.join(output_dir, "plots", filename)
        plt.xlabel("Signed viewing zenith angle [°]")
        plt.xticks([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
        plt.ylabel("BRF [dimensionless]")
        plt.title(fr"$\theta$ = {illumination_zenith}°")
        plt.legend(
            [f"{reflectance}" for reflectance in reflectance_values],
            title=r"$\rho$",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.tight_layout()
        fig.savefig(fname_plot, dpi=200)
        plt.close()

    for illumination_zenith in illumination_zenith_values:
        for reflectance in reflectance_values:
            assert np.allclose(
                results[illumination_zenith][reflectance].brf, reflectance
            )
