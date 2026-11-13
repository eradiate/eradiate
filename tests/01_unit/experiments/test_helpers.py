import pytest

import eradiate
from eradiate.experiments._helpers import (
    atmosphere_converter,
    measure_inside_atmosphere,
    surface_converter,
)
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularComponent,
    ParticleOTComponent,
)
from eradiate.scenes.surface import BasicSurface
from eradiate.util.deprecation import DeprecatedWarning


def test_helpers_measure_inside_atmosphere(
    mode_mono,
    atmosphere_us_standard_mono,
):
    atm = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        components=[atmosphere_us_standard_mono],
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


def test_helpers_atmosphere_converter(mode_mono):
    # Atmospheres pass through
    assert isinstance(
        atmosphere_converter({"type": "homogeneous"}), HomogeneousAtmosphere
    )

    # A single component is wrapped, keeping its ID
    component = ParticleOTComponent(id="aerosols")
    atmosphere = atmosphere_converter(component)
    assert isinstance(atmosphere, HeterogeneousAtmosphere)
    assert atmosphere.id == "aerosols"
    assert isinstance(atmosphere.components[0], ParticleOTComponent)

    # Deprecated medium params are moved to the wrapping atmosphere
    with pytest.warns(DeprecatedWarning, match="'use_rrt','scale'"):
        legacy = ParticleOTComponent(use_rrt=False, scale=2.0)
    for _ in range(2):  # The user's component is left untouched and reusable
        atmosphere = atmosphere_converter(legacy)
        atmosphere.update()
        assert atmosphere.id == "atmosphere"
        assert not atmosphere.use_rrt
        assert atmosphere.scale == 2.0
        assert atmosphere.components[0].use_rrt is None
        assert atmosphere.components[0].scale is None

    # A list of components is wrapped, keeping their order
    atmosphere = atmosphere_converter([{"type": "molecular"}, component])
    assert isinstance(atmosphere, HeterogeneousAtmosphere)
    assert atmosphere.id == "atmosphere"
    assert isinstance(atmosphere.components[0], MolecularComponent)
    assert atmosphere.components[1] is component
