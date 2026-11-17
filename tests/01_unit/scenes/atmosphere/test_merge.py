import attrs
import numpy as np
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.grid import CartesianGrid
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MergeAll,
    MergeByGrid,
    MergedComponent,
    MolecularComponent,
    OverlapAll,
    ParticleOTComponent,
)
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import MultiPhaseFunction
from eradiate.testing.types import check_scene_element

from .test_heterogeneous import default_spectral_index


def cloud_geometry():
    grid = CartesianGrid(
        x_edges=[-1.0, 0.0, 1.0] * ureg.km,
        y_edges=[-1.0, 0.0, 1.0] * ureg.km,
        z_edges=np.linspace(1.0, 3.0, 5) * ureg.km,
    )
    return PlaneParallelGeometry(grid=grid, crop_to_grid="xyz")


def cloud(**kwargs):
    return ParticleOTComponent(
        bottom=1.0 * ureg.km, top=3.0 * ureg.km, geometry=cloud_geometry(), **kwargs
    )


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
    grid = merged.geometry.grid
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
    "policy, expected",
    [
        ("merge_all", [[0, 1, 2]]),
        ("overlap_all", [[0], [1], [2]]),
        ("merge_by_grid", [[0, 1], [2]]),
    ],
)
def test_merge_policy_groups(mode_mono, policy, expected):
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

    for group in atmosphere.components_merged:
        for member in group.components:
            expected_geometry = member._own_geometry or group.geometry
            assert member.geometry is expected_geometry

    if policy == "merge_all":
        assert atmosphere.components_merged[0].geometry is atmosphere.geometry


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


def test_merge_geometry_override_idempotent(mode_mono):
    particle, particle_cloud = ParticleOTComponent(), cloud()
    atmosphere = HeterogeneousAtmosphere(
        components=[particle, particle_cloud], merge_policy="overlap_all"
    )

    # Mimic the experiment geometry override
    geometry = PlaneParallelGeometry(toa_altitude=50.0 * ureg.km)
    atmosphere.geometry = geometry
    atmosphere.update()
    atmosphere.update()

    group, group_cloud = atmosphere.components_merged
    assert len(atmosphere.components_merged) == 2
    assert group.geometry is geometry and particle.geometry is geometry
    assert group_cloud.geometry is particle_cloud._own_geometry
    assert particle.id == "atmosphere_component_0"


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
