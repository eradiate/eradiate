"""Basic CKD correctness check."""

import numpy as np

from eradiate import unit_registry as ureg
from eradiate.experiments import OneDimExperiment
from eradiate.scenes.measure import MultiDistantMeasure
from eradiate.scenes.surface import LambertianSurface


def test_ckd_basic(modes_all_ckd):
    """
    CKD correctness check
    =====================

    We check for correctness when using CKD modes with atmosphere-free scenes.
    This test is designed to check that the CKD infrastructure
    (preprocessing, spectral loop, postprocessing) works.
    """

    # Configure experiment, run and postprocess results
    exp = OneDimExperiment(
        atmosphere=None,
        surface=LambertianSurface(reflectance=1.0),
        measures=[
            MultiDistantMeasure.from_viewing_angles(
                zeniths=np.arange(-60, 61, 5) * ureg.deg,
                azimuths=0.0 * ureg.deg,
                spectral_cfg={
                    "bin_set": "10nm_test",
                    "bins": [
                        "550",
                        "560",
                        "570",
                        "510",
                    ],  # We specify bins in arbitrary order on purpose
                },
            )
        ],
    )
    exp.run()

    # Reflectance is uniform, equal to 1
    assert np.allclose(exp.results["measure"].data_vars["brf"], 1.0)
