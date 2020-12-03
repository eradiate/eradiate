"""A series of test cases the assert the plausibility of Eradiate's computation."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from eradiate.util.plot import remove_xylabels

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
    config["measure"] = [{"type": "toa_pplane_lo",
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

    # Fold negative VZA onto the positive half-space
    results = app.results["toa_pplane_lo"]["lo"]
    lo_zero = np.squeeze(results.where(results.vza >= 0., drop=True).values)[1:]
    lo_pi = np.squeeze(results.where(results.vza < 0., drop=True).values)[::-1]
    vza_values = results.where(results.vza > 0., drop=True).vza.values
    results_postprocessed = xr.Dataset(
        data_vars={
            "lo_zero": ("vza", lo_zero),
            "lo_pi": ("vza", lo_pi),
        },
        coords={
            "vza": ("vza", vza_values)
        }
    )
    results_postprocessed["diff"] = \
        2. * (results_postprocessed["lo_zero"] - results_postprocessed["lo_pi"]) / \
        (results_postprocessed["lo_zero"] + results_postprocessed["lo_pi"])

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(5, 2.5))
    results_postprocessed["lo_zero"].plot(ax=ax1)
    results_postprocessed["lo_pi"].plot(ax=ax1)
    results_postprocessed["diff"].plot(ax=ax2)
    remove_xylabels(from_=[ax1, ax2])
    ax2.yaxis.tick_right()
    plt.suptitle(f"Surface: {surface}, Atmosphere: {str(atmosphere)}")
    plt.tight_layout()

    filename = f"{surface}_{str(atmosphere)}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)

    fig.savefig(fname_plot, dpi=200)
    plt.close()

    assert np.allclose(results_postprocessed["lo_zero"],
                       results_postprocessed["lo_pi"],
                       rtol=5.e-3)
