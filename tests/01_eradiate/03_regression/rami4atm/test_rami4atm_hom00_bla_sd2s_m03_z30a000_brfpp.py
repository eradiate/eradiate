import numpy as np
import pytest

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.test_tools.regression import Chi2Test


@pytest.mark.regression
def test_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(
    mode_ckd_double, artefact_dir, session_timestamp
):
    r"""
    RAMI4ATM HOM00_BLA_SD2S_M03 regression test
    ===========================================

    This regression test uses the ``HOM00_BLA_SD2S_M03_z30a000-brfpp`` scenario
    of the RAMI4ATM benchmark. The reference solution is trusted and compared
    against the libRadtran and RTMOM radiative transfer models.

    Rationale
    ---------
    This test case uses a basic atmospheric scene:

    * black surface
    * Molecular atmosphere following the AFGL 1986 model
    * Aerosol layer at 1km elevation

    Parameters

    * Atmosphere: molecular atmosphere using the AFGL 1986 profile
      (U.S. standard)
    * Aerosol layer: uniform, covers [0, 2] km extent, :math:`\tau_{500} = 0.2`;
      radiative properties from the ``govaerts_2021-desert`` dataset
    * Surface: black
    * Illumination: directional illumination with a zenith angle
      :math:`\theta = 30°`
    * Measure: distant measure, covering the principal plane for
      :math:`\theta \in [-75, 75]°`

    Expected behaviour
    ------------------
    This test uses the Chi-squared criterion with a threshold of 0.05.
    """

    config = {
        "surface": {
            "reflectance": {"value": 0.0, "type": "uniform"},
            "type": "lambertian",
        },
        "atmosphere": {
            "molecular_atmosphere": {
                "has_absorption": False,
                "has_scattering": True,
                "type": "molecular",
                "construct": "afgl_1986",
                "model": "us_standard",
                "levels": [float(i) for i in range(121)],
                "levels_units": "km",
            },
            "particle_layers": [
                {
                    "bottom": 0,
                    "bottom_units": "meter",
                    "top": 2000,
                    "top_units": "meter",
                    "distribution": {"type": "uniform"},
                    "tau_ref": 0.2,
                    "dataset": "govaerts_2021-desert",
                }
            ],
            "type": "heterogeneous",
        },
        "illumination": {
            "zenith": 30.0,
            "zenith_units": "degree",
            "azimuth": 0.0,
            "azimuth_units": "degree",
            "type": "directional",
        },
        "measures": [
            {
                "type": "mdistant",
                "construct": "hplane",
                "zeniths": np.arange(-75, 76, 2),
                "zeniths_units": "degree",
                "azimuth": 0.0,
                "azimuth_units": "degree",
                "srf": "sentinel_2a-msi-3",
                "spp": 1000,
            }
        ],
    }

    exp = AtmosphereExperiment(**config)
    result = eradiate.run(exp)

    test = Chi2Test(
        name=f"{session_timestamp:%Y%m%d-%H%M%S}-rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp",
        value=result,
        reference="tests/regression_test_references/rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp_ref.nc",
        threshold=0.05,
        archive_dir=artefact_dir,
    )

    assert test.run()
