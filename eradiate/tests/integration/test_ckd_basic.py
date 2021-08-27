"""Basic CKD correctness check."""

import numpy as np

from eradiate.scenes.measure import DistantReflectanceMeasure
from eradiate.scenes.surface import LambertianSurface
from eradiate.solvers.onedim import OneDimScene, OneDimSolverApp


def test_ckd_basic(modes_all_ckd):
    """
    CKD correctness check
    =====================

    We check for correctness when using CKD modes with atmosphere-free scenes.
    This test is designed to check that the CKD infrastructure
    (preprocessing, spectral loop, postprocessing) works.
    """

    # Configure scene and app, run solver and postprocess results
    scene = OneDimScene(
        atmosphere=None,
        surface=LambertianSurface(reflectance=1.0),
        measures=[
            DistantReflectanceMeasure(
                spectral_cfg={
                    "bin_set": "10nm_test",
                    "bins": [
                        "550",
                        "560",
                        "570",
                        "510",
                    ],  # We specify bins in arbitrary order on purpose
                }
            )
        ],
    )

    app = OneDimSolverApp(scene)
    app.run()

    # Reflectance is uniform, equal to 1
    assert np.allclose(app.results["measure"].data_vars["brf"], 1.0)
