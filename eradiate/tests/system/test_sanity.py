"""A series of test cases the assert the plausibility of Eradiate's computation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import os

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


@pytest.mark.parametrize("atmosphere", [None, "homogeneous"])
@pytest.mark.parametrize("surface", ["lambertian", "rpv"])
@pytest.mark.slow
def test_symmetry_zenith(variant_scalar_mono, surface, atmosphere):
    r"""
    Result symmetry
    ===============

    This testcase simulates different scenes with illumination at the zenith, asserting
    that the results are symmetrical around that direction.

    The following images display the test result. The left plot in each figure displays the TOA leaving radiance
    plotted against the viewing zenith angle. Both halves of the principal plane are overlaid here. The right
    plot shows the relative difference between the two halves of the principal plane, again plotted against the
    viewing zenith angle.

    .. image:: generated/plots/lambertian_None.png
       :width: 45%
    .. image:: generated/plots/lambertian_homogeneous.png
       :width: 45%

    .. image:: generated/plots/rpv_None.png
       :width: 45%
    .. image:: generated/plots/rpv_homogeneous.png
       :width: 45%

    Rationale
    ---------

        - Geometry: A square surface with either lambertian or RPV BRDF model
        - Illumination: Monochromatic directional illumination from the zenith
        - Atmosphere: Either no atmosphere or a purely rayleigh scattering homogeneous atmosphere
        - Sensor: Principal plane measure, recording TOA leaving radiance with a zenith angular resolution
          of 10° at :math:`10^6` SPP.

    Expected behaviour
    ------------------

        The results on either side of the zenith must agree within a relative tolerance of 5e-3.
    """

    from eradiate.solvers.onedim.app import OneDimSolverApp

    config = {}

    config["mode"] = {"type": "mono",
                      "wavelength": 550.}

    config["illumination"] = {"type": "directional",
                              "zenith": 0.,
                              "azimuth": 0.,
                              "irradiance": {
                                  "type": "uniform",
                                  "value": 1.e6
                              }}
    config["measure"] = [{"type": "toa_lo_pplane",
                          "zenith_res": 10,
                          "spp": 1000000}]

    if surface == "lambertian":
        config["surface"] = {"type": "lambertian",
                             "reflectance": {
                                 "type": "uniform",
                                 "value": 0.5
                             }}
    elif surface == "rpv":
        config["surface"] = {"type": "rpv",
                             "rho_0": 0.183,
                             "k": 0.780,
                             "ttheta": -0.1}

    if atmosphere is None:
        config["atmosphere"] = None
    elif atmosphere == "homogeneous":
        config["atmosphere"] = {"type": "rayleigh_homogeneous",
                                "sigma_s": 1.e-2}

    app = OneDimSolverApp(config)
    app.run()

    results = app.results["toa_lo_pplane"]["toa_lo_pplane"]
    results_zero = np.squeeze(results.ert.sel(vaa=0).values)
    results_pi = np.squeeze(results.ert.sel(vaa=180).values)
    results_diff = []
    for i in range(len(results_zero)):
        if results_zero[i] + results_pi[i] == 0:
            results_diff.append(0)
        else:
            results_diff.append(
                2 * (results_zero[i] - results_pi[i]) /
                (results_zero[i] + results_pi[i])
            )
    ordinate = results.vza.values

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(5, 2.5))
    ax1.plot(ordinate, results_zero)
    ax1.plot(ordinate, results_pi)
    ax2.plot(ordinate, results_diff)
    ax2.yaxis.tick_right()
    plt.suptitle(f"Surface: {surface}, Atmosphere: {str(atmosphere)}")
    plt.tight_layout()

    filename = f"{surface}_{str(atmosphere)}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)

    fig.savefig(fname_plot, dpi=200)
    plt.close()

    assert np.allclose(results_zero, results_pi, rtol=5.e-3)
