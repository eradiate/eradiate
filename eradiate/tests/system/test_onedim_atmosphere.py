"""
Test cases for the one-dimensional solver with a heterogeneous atmosphere.
"""
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg


@pytest.mark.parametrize("bottom", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("tau_550", [0.1, 1.0, 10.0])
def test_heterogeneous_atmosphere_contains_particle_layer(mode_mono, bottom, tau_550):
    """
    HeterogeneousAtmosphere is a good container for a ParticleLayer.

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
    config1 = {"atmosphere": layer}
    app1 = eradiate.solvers.onedim.OneDimSolverApp(config1)
    app1.run()
    results1 = app1.results["measure"].lo.values

    # heterogeneous atmosphere with a particle layer
    config2 = {
        "atmosphere": {"type": "heterogeneous", "particle_layers": [layer]},
    }
    app2 = eradiate.solvers.onedim.OneDimSolverApp(config2)
    app2.run()
    results2 = app2.results["measure"].lo.values

    assert np.all(results1 == results2)


@pytest.mark.parametrize("has_scattering", [True, False])
def test_heterogeneous_atmosphere_contains_molecular_atmosphere(
    mode_mono, has_scattering
):
    """
    HeterogeneouAtmosphere is a good container for a MolecularAtmosphere.

    Same results are produced for a scene consisting of:
       * a non-absorbing molecular atmosphere
       * a heterogeneous atmosphere that (only) contains that non-absorbing
       molecular atmosphere
    """
    # non absorbing molecular atmosphere
    config1 = {
        "atmosphere": {
            "type": "molecular",
            "has_absorption": False,
            "has_scattering": has_scattering,
        }
    }
    app1 = eradiate.solvers.onedim.OneDimSolverApp(config1)
    app1.run()
    results1 = app1.results["measure"].lo.values

    # heterogeneous atmopshere with a non-absorbing molecular atmosphere
    config2 = {
        "atmosphere": {
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "has_absorption": False,
                "has_scattering": has_scattering,
            },
        }
    }
    app2 = eradiate.solvers.onedim.OneDimSolverApp(config2)
    app2.run()
    results2 = app2.results["measure"].lo.values

    assert np.all(results1 == results2)
