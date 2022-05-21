import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.slow
@pytest.mark.parametrize("bottom", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("tau_ref", [0.1, 1.0, 10.0])
def test_heterogeneous_atmosphere_contains_particle_layer(
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

    Run a OneDimExperiment with two different atmosphere definitions:

    1. A single particle layer.
    2. Heterogeneous atmosphere that contains a single particle layer.

    Expected behaviour
    ------------------

    The two experiments produce *exactly* the same results.
    """
    # measure
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "from_viewing_angles",
        "zeniths": np.arange(-90, 91, 15),
        "azimuths": 0.0,
        "spp": 1e4,
        "spectral_cfg": eradiate.scenes.measure.MeasureSpectralConfig.new(
            bin_set="10nm", bins="550"
        ),
    }

    # particle layer
    bottom = bottom * ureg.km
    top = bottom + 1.0 * ureg.km
    layer = eradiate.scenes.atmosphere.ParticleLayer(
        bottom=bottom, top=top, tau_ref=tau_ref
    )
    exp1 = eradiate.experiments.OneDimExperiment(
        atmosphere=layer,
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp1, seed_state=ert_seed_state)
    results1 = exp1.results["measure"]["radiance"].values

    # heterogeneous atmosphere with a particle layer
    exp2 = eradiate.experiments.OneDimExperiment(
        atmosphere={"type": "heterogeneous", "particle_layers": [layer]},
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp2, seed_state=ert_seed_state)
    results2 = exp2.results["measure"]["radiance"].values

    assert np.all(results1 == results2)


@pytest.mark.slow
def test_heterogeneous_atmosphere_contains_molecular_atmosphere(
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

    Run a OneDimExperiment with two different atmosphere definitions:

    1. A molecular atmosphere.
    2. Heterogeneous atmosphere that contains a molecular atmosphere only.

    Expected behaviour
    ------------------

    The two experiments produce *exactly* the same results.
    """
    # measure
    measure = {
        "type": "distant",
        "id": "measure",
        "construct": "from_viewing_angles",
        "zeniths": np.arange(-90, 90, 15),
        "azimuths": 0.0,
        "spp": 1e4,
        "spectral_cfg": eradiate.scenes.measure.MeasureSpectralConfig.new(
            bin_set="10nm", bins="550"
        ),
    }

    # non absorbing molecular atmosphere
    exp1 = eradiate.experiments.OneDimExperiment(
        atmosphere={"type": "molecular", "has_absorption": False},
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp1, seed_state=ert_seed_state)
    results1 = exp1.results["measure"]["radiance"].values

    # heterogeneous atmopshere with a non-absorbing molecular atmosphere
    exp2 = eradiate.experiments.OneDimExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {"type": "molecular", "has_absorption": False},
        },
        measures=[measure],
    )
    ert_seed_state.reset()
    eradiate.run(exp2, seed_state=ert_seed_state)
    results2 = exp2.results["measure"]["radiance"].values

    assert np.all(results1 == results2)
