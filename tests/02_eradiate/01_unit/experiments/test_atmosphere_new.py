import numpy as np

from eradiate import unit_registry as ureg
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.atmosphere import (
    HomogeneousAtmosphere,
    MolecularAtmosphere,
)
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


def test_atmosphere_experiment_mono(mode_mono):
    # TODO: add more atmosphere types (ParticleLayer, HeterogeneousAtmosphere)
    exp = AtmosphereExperiment(
        atmosphere=MolecularAtmosphere.ussa_1976(),
        surface={"type": "lambertian"},
        measures={"type": "distant", "id": "distant_measure"},
    )
    exp.init()
    exp.process()


def test_atmosphere_experiment_ckd(mode_ckd):
    """
    AtmosphereExperiment with heterogeneous atmosphere in CKD mode can be created.
    """
    exp = AtmosphereExperiment(
        atmosphere=MolecularAtmosphere.afgl_1986(),
        surface={"type": "lambertian"},
        measures={"type": "distant", "id": "distant_measure"},
    )
    exp.init()
