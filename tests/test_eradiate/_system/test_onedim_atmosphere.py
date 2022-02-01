import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.parametrize("bottom", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("tau_550", [0.1, 1.0, 10.0])
def test_heterogeneous_atmosphere_contains_particle_layer(
    mode_mono, bottom, tau_550, ert_seed_state
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
    Run a OneDimExperiment with two different atmosphere definitions:

    1. A single particle layer.
    2. Heterogeneous atmosphere that contains a single particle layer.

    Expected behaviour
    ------------------
    The two experiments produce *exactly* the same results.
    """
    # particle layer
    bottom = bottom * ureg.km
    top = bottom + 1.0 * ureg.km
    layer = eradiate.scenes.atmosphere.ParticleLayer(
        bottom=bottom, top=top, tau_550=tau_550
    )
    exp1 = eradiate.experiments.OneDimExperiment(atmosphere=layer)
    ert_seed_state.reset()
    exp1.run(seed_state=ert_seed_state)
    results1 = exp1.results["measure"]["radiance"].values

    # heterogeneous atmosphere with a particle layer
    exp2 = eradiate.experiments.OneDimExperiment(
        atmosphere={"type": "heterogeneous", "particle_layers": [layer]}
    )
    ert_seed_state.reset()
    exp2.run(seed_state=ert_seed_state)
    results2 = exp2.results["measure"]["radiance"].values

    assert np.all(results1 == results2)


def test_heterogeneous_atmosphere_contains_molecular_atmosphere(
    mode_mono_double, ert_seed_state
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
    Run a OneDimExperiment with two different atmosphere definitions:

    1. A molecular atmosphere.
    2. Heterogeneous atmosphere that contains a molecular atmosphere only.

    Expected behaviour
    ------------------
    The two experiments produce *exactly* the same results.
    """
    # non absorbing molecular atmosphere
    exp1 = eradiate.experiments.OneDimExperiment(
        atmosphere={"type": "molecular", "has_absorption": False}
    )
    ert_seed_state.reset()
    exp1.run(seed_state=ert_seed_state)
    results1 = exp1.results["measure"]["radiance"].values

    # heterogeneous atmopshere with a non-absorbing molecular atmosphere
    exp2 = eradiate.experiments.OneDimExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"type": "molecular", "has_absorption": False},
        }
    )
    ert_seed_state.reset()
    exp2.run(seed_state=ert_seed_state)
    results2 = exp2.results["measure"]["radiance"].values

    assert np.all(results1 == results2)
