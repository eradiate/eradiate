"""A series of test cases the assert the plausibility of Eradiate's computation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.plot import remove_xylabels


@pytest.mark.parametrize("atmosphere", ["none", "homogeneous"])
@pytest.mark.parametrize("surface", ["lambertian", "rpv"])
@pytest.mark.slow
def test_symmetry_zenith(
    mode_mono_double, surface, atmosphere, artefact_dir, plot_figures
):
    r"""
    Result symmetry
    ===============

    This test case checks for result symmetry when using the
    ``AtmosphereExperiment`` workflow.

    Rationale
    ---------

    We use a scene with rotational symmetry where the directional illumination
    targets the zenith direction.

    * Geometry: Either a Lambertian or RPV surface.
    * Illumination: Directional illumination from the zenith (default
      irradiance).
    * Atmosphere: Either no atmosphere or a purely Rayleigh scattering
      homogeneous atmosphere.
    * Sensor: Distant measure covering a plane (17 angular points, :math:`10^6`
      SPP) and targeting (0, 0, 0).
      We take an odd number of angular points to ensure that the special value
      :math:`\theta = 0°` is included.

    Expected behaviour
    ------------------

    The results on either side of the zenith must agree within a relative
    tolerance of 5e-3.

    Results
    -------

    For each of the four combinations of atmosphere and surface, two plots are
    shown. On the left of each pair, the recorded leaving radiance is plotted
    against the viewing zenith angle on top of its mirror image for a visual
    inspection of the result symmetry. On the right of each pair the absolute
    difference between these two curves is shown, for a more quantitative
    visualization.

    .. image:: generated/plots/test_symmetry_zenith_lambertian_none.png
       :width: 45%
    .. image:: generated/plots/test_symmetry_zenith_lambertian_homogeneous.png
       :width: 45%

    .. image:: generated/plots/test_symmetry_zenith_rpv_none.png
       :width: 45%
    .. image:: generated/plots/test_symmetry_zenith_rpv_homogeneous.png
       :width: 45%
    """
    spp = int(1e6)
    n_vza = 17

    # Run simulation
    exp = eradiate.experiments.AtmosphereExperiment(
        geometry={"type": "plane_parallel", "toa_altitude": 1.0e2 * ureg.km},
        illumination={"type": "directional", "zenith": 0.0, "azimuth": 0.0},
        measures={
            "type": "mdistant",
            "id": "toa_pplane",
            "construct": "hplane",
            "zeniths": np.linspace(-89, 89, n_vza),
            "azimuth": 0.0,
            "spp": spp,
        },
        surface={
            "lambertian": {"type": "lambertian", "reflectance": 0.5},
            "rpv": {"type": "rpv", "rho_0": 0.183, "k": 0.780, "g": -0.1},
        }[surface],
        atmosphere={
            "none": None,
            "homogeneous": {
                "type": "homogeneous",
                "sigma_s": 1.0e-2 * ureg.km**-1,
            },
        }[atmosphere],
    )
    results = eradiate.run(exp)

    # Post-process results
    radiance = results["radiance"].squeeze().values
    results["diff"] = (
        results["radiance"].dims,
        (radiance - radiance[::-1]).reshape(results["radiance"].shape),
    )
    results["radiance_reversed"] = (
        results["radiance"].dims,
        radiance[::-1].reshape(results["radiance"].shape),
    )

    # Plot results
    if plot_figures:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
        results["radiance"].plot(ax=ax1, x="vza", marker=".", ls="--")
        results["radiance_reversed"].plot(ax=ax1, x="vza", marker=".", ls="--")

        # Scale the difference by its leading power of ten and carry the
        # exponent in the axis label
        diff_max = np.nanmax(np.abs(results["diff"].values))
        exp = np.ceil(np.log10(diff_max)) if diff_max > 0 else 0.0
        (results["diff"] / 10**exp).plot(ax=ax2, x="vza", marker=".", ls="--")

        remove_xylabels(from_=[ax1, ax2])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(f"×$10^{{{int(exp)}}}$")
        ax1.set_title("")
        ax2.set_title("")
        fig.suptitle(f"Surface: {surface}, Atmosphere: {str(atmosphere)}")

        filename = f"test_symmetry_zenith_{surface}_{str(atmosphere)}.png"
        outdir = Path(artefact_dir) / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        fname_plot = outdir / filename

        fig.savefig(fname_plot, dpi=200, bbox_inches="tight")
        plt.close()

    # Check symmetry
    assert np.allclose(
        results["radiance"].squeeze(),
        results["radiance"].squeeze().loc[::-1].data,
        rtol=5e-3,
    )
