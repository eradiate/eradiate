import attrs
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.experiments import AtmosphereExperiment
from eradiate.grid import CartesianGrid, SphericalGrid
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MergeAll,
    MergeByGrid,
    MergedComponent,
    MolecularComponent,
    OverlapAll,
    ParticleOTComponent,
)
from eradiate.scenes.atmosphere._core import geometry_on_grid
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry
from eradiate.scenes.phase import MultiPhaseFunction
from eradiate.testing.types import check_scene_element
from eradiate.util.deprecation import DeprecatedWarning

from .test_heterogeneous import default_spectral_index


def cloud_grid():
    return CartesianGrid(
        x_edges=[-1.0, 0.0, 1.0] * ureg.km,
        y_edges=[-1.0, 0.0, 1.0] * ureg.km,
        z_edges=np.linspace(1.0, 3.0, 5) * ureg.km,
    )


def cloud(**kwargs):
    return ParticleOTComponent(
        bottom=1.0 * ureg.km,
        top=3.0 * ureg.km,
        grid=cloud_grid(),
        crop_to_grid="xyz",
        **kwargs,
    )


def test_geometry_on_grid(mode_mono):
    frame = PlaneParallelGeometry(toa_altitude=50.0 * ureg.km)
    assert geometry_on_grid(frame, None) is frame

    grid = cloud_grid()
    geometry = geometry_on_grid(frame, grid, "xyz")
    assert geometry.grid is grid and geometry.crop_to_grid == "xyz"
    assert geometry.width == frame.width
    assert geometry.ground_altitude == 1.0 * ureg.km
    assert geometry.toa_altitude == 3.0 * ureg.km

    with pytest.raises(ValueError, match="does not match"):
        geometry_on_grid(SphericalShellGeometry(), grid)
    with pytest.raises(ValueError, match="does not match"):
        geometry_on_grid(frame, SphericalGrid.from_levels([0.0, 1.0, 2.0] * ureg.km))


def test_component_grid(mode_mono):
    with pytest.raises(TypeError, match="CartesianGrid"):
        ParticleOTComponent(grid=[0.0, 1.0] * ureg.km)

    component = ParticleOTComponent(
        grid={
            "type": "cartesian",
            "x_edges": [-1.0, 1.0] * ureg.km,
            "y_edges": [-1.0, 1.0] * ureg.km,
            "z_edges": [0.0, 1.0] * ureg.km,
        }
    )
    assert isinstance(component.grid, CartesianGrid)

    with pytest.warns(UserWarning, match="'grid' is unset"):
        ParticleOTComponent(crop_to_grid="z")

    grid = SphericalGrid.from_levels([0.0, 1.0, 2.0] * ureg.km)
    assert isinstance(
        MolecularComponent(grid=grid).default_geometry, SphericalShellGeometry
    )


def test_component_geometry_deprecated(mode_mono):
    geometry = PlaneParallelGeometry(grid=cloud_grid(), crop_to_grid="xyz")
    with pytest.warns(DeprecatedWarning, match="'grid' and 'crop_to_grid'"):
        component = ParticleOTComponent(geometry=geometry)

    assert component.geometry is None
    assert component.grid is geometry.grid
    assert component.crop_to_grid == "xyz"


def test_merged_component_validation(mode_mono):
    with pytest.raises(ValueError, match="at least one component"):
        MergedComponent(components=[])

    nested = MergedComponent(components=[ParticleOTComponent()])
    with pytest.raises(ValueError, match="cannot be nested"):
        MergedComponent(components=[nested])


def test_merged_component_radprops(mode_mono):
    molecular = MolecularComponent()
    particle = ParticleOTComponent()
    merged = MergedComponent(components=[molecular, particle])
    si = KernelContext().si
    grid = merged.default_geometry.grid
    units = ucc.get("collision_coefficient")

    for name in ["sigma_t", "sigma_s"]:
        expected = sum(
            np.broadcast_to(
                getattr(c, f"eval_{name}")(si, grid).m_as(units), grid.shape
            )
            for c in [molecular, particle]
        )
        result = getattr(merged, f"eval_{name}")(si).m_as(units)
        np.testing.assert_allclose(result, expected)

    assert merged.absorption_data is molecular.absorption_data
    assert isinstance(merged.phase, MultiPhaseFunction)
    assert MergedComponent(components=[particle]).phase is particle.phase


def test_merge_policy_conversion(mode_mono):
    components = [ParticleOTComponent()]
    for spec, cls in [
        ("merge_all", MergeAll),
        ({"type": "overlap_all"}, OverlapAll),
        (MergeByGrid(), MergeByGrid),
    ]:
        atmosphere = HeterogeneousAtmosphere(components=components, merge_policy=spec)
        assert isinstance(atmosphere.merge_policy, cls)


@pytest.mark.parametrize(
    "policy, expected, expected_cloud_groups",
    [
        ("merge_all", [[0, 1, 2]], []),
        ("overlap_all", [[0], [1], [2]], [2]),
        ("merge_by_grid", [[0, 1], [2]], [1]),
    ],
)
def test_merge_policy_groups(mode_mono, policy, expected, expected_cloud_groups):
    components = [MolecularComponent(), ParticleOTComponent(), cloud()]
    atmosphere = HeterogeneousAtmosphere(components=components, merge_policy=policy)

    groups = [
        [components.index(member) for member in group.components]
        for group in atmosphere.components_merged
    ]
    assert groups == expected
    assert [group.id for group in atmosphere.components_merged] == [
        f"atmosphere_group_{i}" for i in range(len(expected))
    ]

    # Groups are rendered on the cloud grid or on the atmosphere geometry
    for i, group in enumerate(atmosphere.components_merged):
        if i in expected_cloud_groups:
            assert group.grid is components[2].grid
            assert group.geometry.grid is components[2].grid
            assert group.geometry.crop_to_grid == "xyz"
        else:
            assert group.grid is None
            assert group.geometry is atmosphere.geometry


def test_merge_extremum_resolution(mode_mono):
    components = [
        ParticleOTComponent(extremum_resolution=(1, 1, 4)),
        ParticleOTComponent(),
    ]

    atmosphere = HeterogeneousAtmosphere(
        components=components, merge_policy="overlap_all"
    )
    resolutions = [g.extremum_resolution for g in atmosphere.components_merged]
    assert resolutions == [(1, 1, 4), None]

    with pytest.warns(UserWarning, match="different extremum resolutions"):
        atmosphere = HeterogeneousAtmosphere(components=components)
    assert atmosphere.components_merged[0].extremum_resolution is None


def test_merge_geometry_override(mode_mono):
    molecular, particle_cloud = MolecularComponent(), cloud()
    atmosphere = HeterogeneousAtmosphere(
        components=[molecular, particle_cloud], merge_policy="overlap_all"
    )

    # Mimic the experiment geometry override
    for geometry in [
        PlaneParallelGeometry(toa_altitude=50.0 * ureg.km, width=10.0 * ureg.km),
        PlaneParallelGeometry(toa_altitude=80.0 * ureg.km),
    ]:
        atmosphere.geometry = geometry
        atmosphere.update()

        group, group_cloud = atmosphere.components_merged
        assert group.geometry is geometry and molecular.geometry is None
        assert group_cloud.geometry.grid is particle_cloud.grid
        assert group_cloud.geometry.width == geometry.width
        assert group_cloud.geometry.toa_altitude == 3.0 * ureg.km
        assert molecular.id == "atmosphere_component_0"

        # Phase functions of components without a grid use the group geometry
        assert group.get_phase(geometry).geometry is geometry

    # Frame type change: the cloud grid no longer fits
    atmosphere.geometry = SphericalShellGeometry()
    with pytest.raises(ValueError, match="does not match"):
        atmosphere.update()


def test_merge_member_grid(mode_mono):
    grid = CartesianGrid.from_levels(np.linspace(0.0, 10.0, 11) * ureg.km)
    molecular = MolecularComponent(grid=grid)
    atmosphere = HeterogeneousAtmosphere(components=[molecular, ParticleOTComponent()])

    # The group is rendered on the frame; the member phase uses its own grid
    (group,) = atmosphere.components_merged
    assert group.geometry is atmosphere.geometry
    phase = group.get_phase(group.geometry)
    assert phase.components[0].geometry.grid is grid


def test_merge_user_groups(mode_mono):
    molecular = MolecularComponent()
    particles = [ParticleOTComponent(), ParticleOTComponent()]
    user_group = MergedComponent(components=particles, extremum_resolution=(1, 1, 2))
    atmosphere = HeterogeneousAtmosphere(
        components=[molecular, user_group], merge_policy="overlap_all"
    )

    group_molecular, group_particles = atmosphere.components_merged
    assert group_molecular.components == [molecular]
    assert group_particles.components == particles
    assert group_particles.extremum_resolution == (1, 1, 2)
    assert group_particles.geometry is atmosphere.geometry
    assert atmosphere.components == [molecular, *particles]
    assert atmosphere.molecular_atmosphere is molecular
    assert atmosphere.particle_components == particles

    # Copies keep user groups, as done by experiments
    evolved = attrs.evolve(atmosphere, geometry=PlaneParallelGeometry())
    assert evolved.components_merged[1].components == particles

    # Dictionary specification
    atmosphere = HeterogeneousAtmosphere(
        components=[{"type": "merged", "components": [{"type": "particle_ot"}]}]
    )
    assert isinstance(atmosphere.components_merged[0], MergedComponent)

    # User group with its own grid
    grid = cloud_grid()
    atmosphere = HeterogeneousAtmosphere(
        components=[
            molecular,
            MergedComponent(components=[cloud()], grid=grid, crop_to_grid="xyz"),
        ]
    )
    group = atmosphere.components_merged[1]
    assert group.geometry.grid is grid and group.geometry.crop_to_grid == "xyz"


def test_merge_duplicate_component(mode_mono):
    particle = ParticleOTComponent()
    with pytest.raises(ValueError, match="only once"):
        HeterogeneousAtmosphere(
            components=[particle, MergedComponent(components=[particle])]
        )


def test_merge_single_group_template(mode_mono):
    atmosphere = HeterogeneousAtmosphere(
        components=[MolecularComponent(), ParticleOTComponent()]
    )
    template, _ = traverse(atmosphere)

    assert not atmosphere.is_overlapping
    assert template["medium_atmosphere.type"] == "piecewise"
    assert template["medium_atmosphere.phase.id"] == "phase_atmosphere"
    assert template["phase_atmosphere.type"] == "multiphase"


def test_merge_overlap_template(mode_mono):
    atmosphere = HeterogeneousAtmosphere(
        components=[MolecularComponent(), cloud(extremum_resolution=(2, 2, 2))],
        merge_policy="overlap_all",
    )
    template, _ = traverse(atmosphere)

    assert atmosphere.is_overlapping
    assert not atmosphere.is_piecewise
    assert template["medium_atmosphere.type"] == "overlap"

    for i, group in enumerate(atmosphere.components_merged):
        prefix = f"medium_atmosphere.component{i}"
        assert template[f"{prefix}.type"] == "eoheterogeneous"
        assert template[f"{prefix}.id"] == group.medium_id
        assert template[f"{prefix}.phase.id"] == group.phase_id
        assert f"{group.phase_id}.type" in template

    assert template["medium_atmosphere.component1.extremum.resolution"] == (2, 2, 2)

    length_units = uck.get("length")
    np.testing.assert_allclose(
        template["medium_atmosphere.component1.aabb_min"],
        ([-1.0, -1.0, 1.0] * ureg.km).m_as(length_units),
    )
    np.testing.assert_allclose(
        template["medium_atmosphere.component1.aabb_max"],
        ([1.0, 1.0, 3.0] * ureg.km).m_as(length_units),
    )

    check_scene_element(atmosphere, ctx=KernelContext())


def test_merge_overlap_ckd(mode_ckd, atmosphere_us_standard_ckd):
    atmosphere = HeterogeneousAtmosphere(
        components=[atmosphere_us_standard_ckd, cloud()],
        merge_policy="merge_by_grid",
    )
    assert atmosphere.is_overlapping
    si = default_spectral_index(atmosphere)
    check_scene_element(atmosphere, ctx=KernelContext(si=si))


def test_merge_spherical_polarized(mode_mono_polarized):
    # The molecular component has no grid: its phase uses the experiment frame
    exp = AtmosphereExperiment(
        geometry="spherical_shell", atmosphere={"type": "molecular"}
    )
    _, params = traverse(exp.atmosphere)
    assert "phase_atmosphere.depolarization.volume.data" in params.data


def test_check_geometry(mode_mono):
    thermoprops = {
        "identifier": "afgl_1986-us_standard",
        "z": np.linspace(0.0, 80.0, 41) * ureg.km,
        "additional_molecules": False,
    }

    # Not checked at construction, on the default 120 km frame
    atmosphere = HeterogeneousAtmosphere(
        components=[MolecularComponent(thermoprops=thermoprops)]
    )
    with pytest.raises(ValueError, match="go beyond"):
        atmosphere.check_geometry()

    # Checked by experiments once the frame is set
    AtmosphereExperiment(
        geometry={"type": "plane_parallel", "toa_altitude": 80.0 * ureg.km},
        atmosphere=atmosphere,
    )
    with pytest.raises(ValueError, match="go beyond"):
        AtmosphereExperiment(atmosphere=atmosphere)

    # The own grid of a member is checked, too
    grid = CartesianGrid.from_levels(np.linspace(0.0, 100.0, 11) * ureg.km)
    atmosphere = HeterogeneousAtmosphere(
        geometry={"type": "plane_parallel", "toa_altitude": 80.0 * ureg.km},
        components=[MolecularComponent(thermoprops=thermoprops, grid=grid)],
    )
    with pytest.raises(ValueError, match="go beyond"):
        atmosphere.check_geometry()
