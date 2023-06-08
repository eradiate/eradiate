import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.parametrize(
    "atm_flags",
    [
        {"molecular": {}},
        {"molecular": {"has_scattering": False}},
        {"molecular": {"has_absorption": False}},
    ],
    ids=["default", "mol_nosc", "mol_noab"],
)
def test_heterogeneous_atm_flags(modes_all_double, atm_flags):
    """
    Heterogeneous atmosphere component flags
    ========================================

    This system test tries various combinations of atmosphere flags and checks
    that a computation succeeds and returns non-zero values.
    """
    # TODO: Recycle this test
    if eradiate.mode().is_mono:
        atm_kwargs = {"construct": "ussa_1976"}
    elif eradiate.mode().is_ckd:
        atm_kwargs = {"construct": "afgl_1986"}
    else:
        raise NotImplementedError

    exp = eradiate.experiments.AtmosphereExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                **atm_kwargs,
                **atm_flags["molecular"],
            },
            "particle_layers": {
                "tau_ref": 0.2,
                "bottom": 0.0,
                "top": 10.0 * ureg.km,
            },
        },
        illumination={
            "type": "directional",
            "zenith": 30.0,
            "azimuth": 0.0,
        },
        measures={
            "type": "mdistant",
            "construct": "hplane",
            "azimuth": 0.0,
            "zeniths": np.arange(-75, 76, 5),
            "srf": eradiate.scenes.spectra.MultiDeltaSpectrum(
                wavelengths=550.0 * ureg.nm
            ),
        },
    )

    assert np.all(eradiate.run(exp) > 0.0)
