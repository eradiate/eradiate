import pytest

import eradiate
from eradiate.experiments._helpers import measure_inside_atmosphere, surface_converter
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
)
from eradiate.scenes.surface import BasicSurface


def test_helpers_measure_inside_atmosphere(mode_mono):
    atm = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=MolecularAtmosphere.afgl_1986(),
    )

    s1 = eradiate.scenes.measure.MultiDistantMeasure()
    assert not measure_inside_atmosphere(atm, s1)

    s2 = eradiate.scenes.measure.PerspectiveCameraMeasure(
        origin=[1, 1, 1], target=[0, 0, 0]
    )
    assert measure_inside_atmosphere(atm, s2)

    s3 = eradiate.scenes.measure.MultiRadiancemeterMeasure(
        origins=[[1, 1, 1], [0, 0, 1000000]], directions=[[0, 0, -1], [0, 0, -1]]
    )
    with pytest.raises(ValueError):
        measure_inside_atmosphere(atm, s3)


def test_helpers_surface_converter(mode_mono):
    # A dictionary specifying a surface is converted to a surface
    assert isinstance(surface_converter({"type": "basic"}), BasicSurface)

    # A dictionary specifying a BSDF is converted to a surface
    assert isinstance(surface_converter({"type": "lambertian"}), BasicSurface)
