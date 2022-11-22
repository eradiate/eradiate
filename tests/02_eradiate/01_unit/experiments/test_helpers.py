import pytest

import eradiate
from eradiate.contexts import KernelDictContext
from eradiate.experiments._helpers import measure_inside_atmosphere
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
)


def test_helpers_measure_inside_atmosphere(mode_mono):
    ctx = KernelDictContext()
    atm = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=MolecularAtmosphere.afgl_1986(),
    )

    s1 = eradiate.scenes.measure.MultiDistantMeasure()
    assert not measure_inside_atmosphere(atm, s1, ctx)

    s2 = eradiate.scenes.measure.PerspectiveCameraMeasure(
        origin=[1, 1, 1], target=[0, 0, 0]
    )
    assert measure_inside_atmosphere(atm, s2, ctx)

    s3 = eradiate.scenes.measure.MultiRadiancemeterMeasure(
        origins=[[1, 1, 1], [0, 0, 1000000]], directions=[[0, 0, -1], [0, 0, -1]]
    )
    with pytest.raises(ValueError):
        measure_inside_atmosphere(atm, s3, ctx)
