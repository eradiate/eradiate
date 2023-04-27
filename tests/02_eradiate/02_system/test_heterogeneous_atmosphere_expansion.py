import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.slow
@pytest.mark.parametrize("bottom", [0.0, 1.0, 1.0])
@pytest.mark.parametrize("tau_ref", [0.1, 1.0, 10.0])
def test_heterogeneous_atmosphere_expansion_particle_layer(
    mode_ckd_double, bottom, tau_ref, ert_seed_state
):
    """
    Perfect single-component HeterogeneousAtmosphere expansion (particle layer)
    ===========================================================================

    This test case checks if a single-component HeterogeneousAtmosphere
    container expands perfectly as its component. This is assessed by checking
    if the results provided by two scenes presumably identical are indeed the
    same.

    Rationale
    ---------

    Run an AtmosphereExperiment with two different atmosphere definitions:

    1. A single particle layer.
    2. Heterogeneous atmosphere that contains a single particle layer.

    Expected behaviour
    ------------------

    The two experiments produce *exactly* the same results.
    """
    spp = 1000

    # Measure
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "hplane",
        "zeniths": np.arange(-90, 91, 15),
        "azimuth": 0.0,
        "srf": eradiate.scenes.spectra.MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
    }

    # Particle layer only
    bottom = bottom * ureg.km
    top = bottom + 1.0 * ureg.km
    geometry = {
        "type": "plane_parallel",
        "ground_altitude": bottom,
        "toa_altitude": top,
    }
    layer = {
        "type": "particle_layer",
        "bottom": bottom,
        "top": top,
        "tau_ref": tau_ref,
    }
    exp1 = eradiate.experiments.AtmosphereExperiment(
        geometry=geometry,
        atmosphere=layer,
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp1, seed_state=ert_seed_state, spp=spp)
    results1 = exp1.results["measure"]["radiance"].values

    # Heterogeneous atmosphere with a particle layer
    exp2 = eradiate.experiments.AtmosphereExperiment(
        geometry=geometry,
        atmosphere={
            "type": "heterogeneous",
            "particle_layers": [layer],
        },
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp2, seed_state=ert_seed_state, spp=spp)
    results2 = exp2.results["measure"]["radiance"].values

    np.testing.assert_equal(results1, results2)


@pytest.mark.slow
def test_heterogeneous_atmosphere_expansion_molecular_atmosphere(
    mode_ckd_double, ert_seed_state
):
    """
    Perfect single-component HeterogeneousAtmosphere expansion (molecular atmosphere)
    =================================================================================

    This test case checks if a single-component HeterogeneousAtmosphere
    container expands perfectly as its component. This is assessed by checking
    if the results provided by two scenes presumably identical are indeed the
    same.

    Rationale
    ---------

    Run an AtmosphereExperiment with two different atmosphere definitions:

    1. A molecular atmosphere.
    2. Heterogeneous atmosphere that contains a molecular atmosphere only.

    Expected behaviour
    ------------------

    The two experiments produce *exactly* the same results.
    """
    spp = 1000

    # measure
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "hplane",
        "zeniths": np.arange(-90, 90, 15),
        "azimuth": 0.0,
        "srf": eradiate.scenes.spectra.MultiDeltaSpectrum(wavelengths=550.0 * ureg.nm),
    }

    # Non-absorbing molecular atmosphere
    exp1 = eradiate.experiments.AtmosphereExperiment(
        atmosphere={"type": "molecular", "has_absorption": False},
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp1, seed_state=ert_seed_state, spp=spp)
    results1 = exp1.results["measure"]["radiance"].values

    # Heterogeneous atmopshere with a non-absorbing molecular atmosphere
    exp2 = eradiate.experiments.AtmosphereExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"type": "molecular", "has_absorption": False},
        },
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp2, seed_state=ert_seed_state)
    results2 = exp2.results["measure"]["radiance"].values

    np.testing.assert_equal(results1, results2)
