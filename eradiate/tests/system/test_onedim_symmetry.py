"""A series of test cases the assert the plausibility of Eradiate's computation."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate.plot import remove_xylabels

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


@pytest.mark.parametrize("atmosphere", ["none", "homogeneous"])
@pytest.mark.parametrize("surface", ["lambertian", "rpv"])
@pytest.mark.slow
def test_symmetry_zenith(mode_mono_double, surface, atmosphere):
    r"""
    Result symmetry
    ===============

    This test case checks for result symmetry when using the ``OneDimSolverApp`` workflow.

    Rationale
    ---------

    We use a scene with rotational symmetry where the directional illumination targets the
    zenith direction.

    * Geometry: Either a Lambertian or RPV surface.
    * Illumination: Directional illumination from the zenith (default irradiance).
    * Atmosphere: Either no atmosphere or a purely Rayleigh scattering homogeneous atmosphere.
    * Sensor: Distant measure covering a plane (17 angular points, :math:`10^6` SPP) and
      targeting (0, 0, 0).
      We take an odd number of angular points to ensure that the special value
      :math:`\theta = 0°` is included.

    Expected behaviour
    ------------------

    The results on either side of the zenith must agree within a relative tolerance of 5e-3.

    Results
    -------

    For each of the four combinations of atmosphere and surface, two plots are
    shown. On the left of each pair, the recorded leaving radiance is plotted
    against the viewing zenith angle on top of its mirror image for a visual
    inspection of the result symmetry. On the right of each pair the absolute
    difference between these two curves is shown, for a more quantitative visualization.

    .. image:: generated/plots/lambertian_none.png
       :width: 45%
    .. image:: generated/plots/lambertian_homogeneous.png
       :width: 45%

    .. image:: generated/plots/rpv_none.png
       :width: 45%
    .. image:: generated/plots/rpv_homogeneous.png
       :width: 45%
    """
    spp = int(1e6)
    n_vza = 17

    scene = eradiate.solvers.onedim.OneDimScene(
        illumination={"type": "directional", "zenith": 0.0, "azimuth": 0.0},
        measures={
            "type": "distant_reflectance",
            "id": "toa_pplane",
            "film_resolution": (n_vza, 1),
            "spp": spp,
        },
        surface={
            "lambertian": {"type": "lambertian", "reflectance": 0.5},
            "rpv": {"type": "rpv", "rho_0": 0.183, "k": 0.780, "g": -0.1},
        }[surface],
        atmosphere={
            "none": None,
            "homogeneous": {"type": "homogeneous", "sigma_s": 1.0e-2},
        }[atmosphere],
    )

    # Run simulation
    app = eradiate.solvers.onedim.OneDimSolverApp(scene=scene)
    app.run()

    # Post-process results
    results = app.results["toa_pplane"]
    results["diff"] = (
        results["lo"].dims,
        (
            results.lo
            - results.lo.squeeze().loc[::-1].data.reshape(results["lo"].shape)
        ).data,
    )
    results["lo_reversed"] = (
        results["lo"].dims,
        results.lo.squeeze().loc[::-1].data.reshape(results["lo"].shape).data,
    )

    # Plot results
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(5, 2.5))
    results["lo"].plot(ax=ax1, x="vza")
    results["lo_reversed"].plot(ax=ax1, x="vza")
    results["diff"].plot(ax=ax2, x="vza")
    remove_xylabels(from_=[ax1, ax2])
    ax2.yaxis.tick_right()
    ax1.set_title("")
    ax2.set_title("")
    plt.suptitle(f"Surface: {surface}, Atmosphere: {str(atmosphere)}")
    plt.tight_layout()

    filename = f"{surface}_{str(atmosphere)}.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)

    fig.savefig(fname_plot, dpi=200)
    plt.close()

    # Check symmetry
    assert np.allclose(
        results["lo"].squeeze(), results["lo"].squeeze().loc[::-1].data, rtol=5e-3
    )
