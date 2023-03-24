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
    This system test tries various combinations of atmosphere flags and checks
    that a computation succeeds and returns non-zero values.
    """
    exp = eradiate.experiments.CanopyAtmosphereExperiment(
        surface={"type": "rpv"},
        atmosphere={
            "type": "heterogeneous",
            "zgrid": np.linspace(0, 120, 13) * ureg.km,
            "molecular_atmosphere": {
                "type": "molecular",
                "construct": "afgl_1986" if eradiate.mode().is_ckd else "ussa_1976",
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
            "spectral_cfg": {"bins": ["550"]},
        },
    )

    assert np.all(eradiate.run(exp) > 0.0)
