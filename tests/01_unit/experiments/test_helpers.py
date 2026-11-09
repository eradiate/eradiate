import pytest

import eradiate
from eradiate.experiments._helpers import (
    atmosphere_converter,
    measure_inside_atmosphere,
    surface_converter,
)
from eradiate.scenes.atmosphere import (
    Atmosphere,
    HomogeneousComponent,
    MolecularComponent,
    ParticleOTComponent,
)
from eradiate.scenes.surface import BasicSurface
from eradiate.util.deprecation import DeprecatedWarning


def test_helpers_measure_inside_atmosphere(
    mode_mono,
    atmosphere_us_standard_mono,
):
    atm = Atmosphere(
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
    # Atmospheres pass through, the 'type' entry defaults to 'atmosphere'
    for spec in [
        {"type": "atmosphere", "components": [{"type": "homogeneous"}]},
        {"components": [{"type": "homogeneous"}]},
    ]:
        atmosphere = atmosphere_converter(spec)
        assert isinstance(atmosphere, Atmosphere)
        assert isinstance(atmosphere.components[0], HomogeneousComponent)

    # A legacy homogeneous atmosphere is wrapped, its medium options are ignored
    with pytest.warns(DeprecatedWarning, match="'HomogeneousComponent.use_rrt'"):
        atmosphere = atmosphere_converter({"type": "homogeneous", "use_rrt": False})
    assert isinstance(atmosphere, Atmosphere)
    assert atmosphere.use_rrt
    assert isinstance(atmosphere.components[0], HomogeneousComponent)

    # A single component is wrapped, keeping its ID
    component = ParticleOTComponent(id="aerosols")
    atmosphere = atmosphere_converter(component)
    assert isinstance(atmosphere, Atmosphere)
    assert atmosphere.id == "aerosols"
    assert isinstance(atmosphere.components[0], ParticleOTComponent)

    # A list of components is wrapped, keeping their order
    atmosphere = atmosphere_converter([{"type": "molecular"}, component])
    assert isinstance(atmosphere, Atmosphere)
    assert atmosphere.id == "atmosphere"
    assert isinstance(atmosphere.components[0], MolecularComponent)
    assert atmosphere.components[1] is component
