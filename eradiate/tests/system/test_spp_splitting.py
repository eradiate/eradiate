import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eradiate
from eradiate import unit_registry as ureg

eradiate_dir = os.environ["ERADIATE_DIR"]
output_dir = os.path.join(eradiate_dir, "test_report", "generated")


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def test_spp_splitting(mode_mono):
    """
    SPP splitting test
    ==================

    We check that SPP splitting applies

    Rationale
    ---------

    * Geometry: a square surface with unit size and a Lambertian BRDF with unit reflectance.
    * Illumination: a directional light source at the zenith with unit irradiance.
    * Measure: A ``distant`` sensor targeting (0, 0, 0) with 1x1 film size.

    We compute the BRF for a Cartesian product of the following configurations:

    * SPP splitting thresholds: 1e5 (default behaviour), 1e10 (disable SPP splitting);
    * operational mode: monochromatic single-precision, monochromatic double-precision;
    * SPP: a series of values following a geometric progression around a central value of
      1e5.

    Expected behaviour
    ------------------

    * Double-precision configurations always yield a correct BRF value of 1.
    * Single-precision configurations yield correct results for SPP <= 1e5.
    * Single-precision configurations yield wrong results without SPP splitting for SPP > 1e5.
    * Single-precision configurations yield correct results with SPP splitting for SPP > 1e5.

    Results
    -------

    .. image:: generated/plots/test_spp_splitting.png
       :width: 66%
    """
    surface = eradiate.scenes.surface.LambertianSurface(
        width=1.0 * ureg.m, reflectance=1.0
    )
    measure = eradiate.scenes.measure.DistantReflectanceMeasure(
        target=[0, 0, 0],
        film_resolution=(1, 1),
    )
    illumination = eradiate.scenes.illumination.DirectionalIllumination(irradiance=1.0)

    scene = eradiate.solvers.onedim.OneDimScene(
        atmosphere=None,
        surface=surface,
        illumination=illumination,
        measures=measure,
    )
    solver = eradiate.solvers.onedim.OneDimSolverApp(scene)

    threshold = 5
    thresholds = [10 ** 10, 10 ** threshold]
    modes = ["mono", "mono_double"]
    spps = [10 ** (threshold - 1), 10 ** threshold, 10 ** (threshold + 1)]
    for spp in copy(spps[:-1]):
        spps.append(2 * spp)
        spps.append(5 * spp)
    spps.sort()
    results = pd.DataFrame(
        index=pd.Index(spps, name="spp"),
        columns=pd.MultiIndex.from_product(
            (thresholds, modes), names=["threshold", "mode"]
        ),
    )

    for spp_splitting_threshold in thresholds:
        solver.scene.measures[0]._spp_splitting_threshold = spp_splitting_threshold

        for mode in modes:
            eradiate.set_mode(mode)

            for spp in spps:
                solver.scene.measures[0].spp = spp
                solver.run()
                results[spp_splitting_threshold, mode][spp] = float(
                    solver.results["measure"].brf
                )

    # Save plot for report
    fig, _ = plt.subplots(1, 2, figsize=(10, 4))
    fig, ax = plt.subplots()
    results.plot(ax=ax, logx=True, marker="o", linestyle="--")
    plt.ylabel("brf")
    plt.tight_layout()

    filename = "test_spp_splitting.png"
    ensure_output_dir(os.path.join(output_dir, "plots"))
    fname_plot = os.path.join(output_dir, "plots", filename)

    fig.savefig(fname_plot, dpi=200)
    plt.close()

    idx = pd.IndexSlice
    # Double-precision configurations always yield a correct BRF value of 1
    assert np.allclose(results.loc[:, idx[:, "mono_double"]].to_numpy(dtype=float), 1.0)
    # Single-precision configurations yield correct results for SPP <= 1e5
    assert np.allclose(
        results.loc[: 1e5 + 1, idx[:, "mono"]].to_numpy(dtype=float), 1.0, rtol=1e-3
    )
    # Single-precision configurations yield wrong results without SPP splitting for SPP > 1e5
    assert not np.allclose(
        results.loc[1e5 + 1 :, idx[10 ** 10, "mono"]].to_numpy(dtype=float),
        1.0,
        rtol=1e-3,
    )
    # Single-precision configurations to yield correct results with SPP splitting for SPP > 1e5
    assert np.allclose(
        results.loc[1e5 + 1 :, idx[10 ** 5, "mono"]].to_numpy(dtype=float),
        1.0,
        rtol=1e-3,
    )
