import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eradiate
from eradiate import unit_registry as ureg


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


@pytest.mark.slow
def test_spp_splitting(mode_mono, artefact_dir):
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

    * SPP splitting thresholds: 1e5, disabled;
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
    # Basic scene setup
    exp = eradiate.experiments.OneDimExperiment(
        atmosphere=None,
        surface=eradiate.scenes.surface.LambertianSurface(
            width=1.0 * ureg.m, reflectance=1.0
        ),
        illumination=eradiate.scenes.illumination.DirectionalIllumination(
            irradiance=1.0
        ),
        measures=eradiate.scenes.measure.MultiDistantMeasure(target=[0, 0, 0]),
    )

    # Generate test matrix
    modes = ["mono", "mono_double"]
    pivot_threshold = 5
    spp_splits = [None, 10 ** pivot_threshold]
    spp_split_labels = ["none", "1e5"]
    spps = [
        10 ** (pivot_threshold - 1),
        10 ** pivot_threshold,
        10 ** (pivot_threshold + 1),
    ]
    for spp in copy(spps[:-1]):
        spps.append(2 * spp)
        spps.append(5 * spp)
    spps.sort()

    results = pd.DataFrame(
        index=pd.Index(spps, name="spp"),
        columns=pd.MultiIndex.from_product(
            (spp_split_labels, modes), names=["threshold", "mode"]
        ),
    )

    for spp_split, spp_split_label in zip(spp_splits, spp_split_labels):
        exp.measures[0].split_spp = spp_split

        for mode in modes:
            eradiate.set_mode(mode)

            for spp in spps:
                exp.measures[0].spp = spp
                exp.run()
                results[spp_split_label, mode][spp] = float(exp.results["measure"].brf)

    # Save plot for report
    fig, _ = plt.subplots(1, 2, figsize=(10, 4))
    fig, ax = plt.subplots()
    results.plot(ax=ax, logx=True, marker="o", linestyle="--")
    plt.ylabel("brf")
    plt.tight_layout()

    filename = "test_spp_splitting.png"
    ensure_output_dir(os.path.join(artefact_dir, "plots"))
    fname_plot = os.path.join(artefact_dir, "plots", filename)

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
        results.loc[1e5 + 1 :, idx["none", "mono"]].to_numpy(dtype=float),
        1.0,
        rtol=1e-3,
    )
    # Single-precision configurations to yield correct results with SPP splitting for SPP > 1e5
    print(results)
    assert np.allclose(
        results.loc[1e5 + 1 :, idx["1e5", "mono"]].to_numpy(dtype=float),
        1.0,
        rtol=1e-3,
    )
