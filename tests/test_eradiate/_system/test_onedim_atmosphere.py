"""
Test cases for the one-dimensional solver with a heterogeneous atmosphere.
"""
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
    HeterogeneousAtmosphere is a good container for a ParticleLayer
    ===============================================================

    This testcase asserts that the HeterogeneousAtmosphere class can act as a container for
    other atmosphere objects.


    Rationale
    ---------

    Run a OneDimExperiment with two different atmosphere definitions.
    First define an atmosphere that is directly made up from a single particle layer.
    Second define a heterogeneous atmosphere that contains *only* one particle layer.

    Expected behaviour
    ------------------

    Same results are produced for a scene consisting of
    * a particle layer
    * a heterogeneous atmosphere that (only) contains that particle layer.
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
    mode_mono, ert_seed_state
):
    """
    HeterogeneousAtmosphere is a good container for a MolecularAtmosphere
    =====================================================================

    This testcase asserts that the HeterogeneousAtmosphere class can act as a
    container for other atmosphere objects.

    Rationale
    ---------

    Run a OneDimExperiment with two different atmosphere definitions.
    First define an atmosphere that is directly made up from a non-absorbing
    molecular atmosphere.
    Second define a heterogeneous atmosphere that contains *only* a
    non-absorbing molecular atmosphere.

    Expected behaviour
    ------------------

    Same results are produced for a scene consisting of:
       * a non-absorbing molecular atmosphere
       * a heterogeneous atmosphere that (only) contains that non-absorbing
         molecular atmosphere
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
