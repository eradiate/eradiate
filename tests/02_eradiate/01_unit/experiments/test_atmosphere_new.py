import numpy as np

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.measure import MultiDistantMeasure


def test_atmosphere_experiment_construct_default(modes_all_double):
    """
    AtmosphereExperiment initialises with default params in all modes
    """
    assert AtmosphereExperiment()


def test_atmosphere_experiment_construct_measures(modes_all):
    """
    A variety of measure specifications are acceptable
    """
    # Init with a single measure (not wrapped in a sequence)
    assert AtmosphereExperiment(measures=MultiDistantMeasure())

    # Init from a dict-based measure spec
    # -- Correctly wrapped in a sequence
    assert AtmosphereExperiment(measures=[{"type": "distant"}])
    # -- Not wrapped in a sequence
    assert AtmosphereExperiment(measures={"type": "distant"})


def test_atmosphere_experiment_construct_normalize_measures(mode_mono):
    # When setting atmosphere to None, measure target is at ground level
    exp = AtmosphereExperiment(atmosphere=None)
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)

    # When atmosphere is set, measure target is at ground level
    exp = AtmosphereExperiment(atmosphere=HomogeneousAtmosphere(top=100.0 * ureg.km))
    assert np.allclose(exp.measures[0].target.xyz, [0, 0, 0] * ureg.m)


def test_atmosphere_experiment(modes_all_double):
    exp = AtmosphereExperiment(
        atmosphere={
            "type": "heterogeneous",
            "molecular_atmosphere": {
                "type": "molecular",
                "construct": "afgl_1986" if eradiate.mode().is_ckd else "ussa_1976",
            },
            "particle_layers": [{"type": "particle_layer"}],
        },
        surface={"type": "lambertian"},
        measures={
            "type": "distant",
            "id": "distant_measure",
            "spectral_cfg": {"srf": "sentinel_2a-msi-3"}
            if eradiate.mode().is_ckd
            else {"wavelengths": [550.0] * ureg.nm},
        },
    )
    exp.process()
    exp.postprocess()
