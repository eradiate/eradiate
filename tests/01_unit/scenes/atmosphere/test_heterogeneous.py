import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    ParticleEnsemble,
)
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry, SceneGeometry, WrapMode
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def default_spectral_index(atmosphere):
    # This is a bit fragile (API stability is not guaranteed and units are not
    # checked) but it works well in both mono and ckd modes
    wavelengths = atmosphere.absorption_data._spectral_coverage.index.get_level_values(
        1
    ).values
    w = _find_nearest(wavelengths, 550.0)
    kwargs = {"w": w * ureg.nm}
    if eradiate.mode().is_ckd:
        kwargs["g"] = 0.5
    return SpectralIndex.new(**kwargs)


def test_heterogeneous_empty(modes_all_double):
    # Passing no component is not allowed
    with pytest.raises(ValueError):
        HeterogeneousAtmosphere()


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("component", ["molecular", "particle"])
def test_heterogeneous_single_mono(
    mode_mono, geometry, component, atmosphere_us_standard_mono
):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if component == "molecular":
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, molecular_atmosphere=atmosphere_us_standard_mono
        )
        si = default_spectral_index(atmosphere.molecular_atmosphere)
        kernel_context = KernelContext(si=si)

    else:
        component = ParticleEnsemble()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_ensembles=[component]
        )
        kernel_context = KernelContext()

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, ctx=kernel_context)


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("component", ["molecular", "particle"])
def test_heterogeneous_single_ckd(
    mode_ckd, geometry, component, atmosphere_us_standard_ckd
):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if component == "molecular":
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, molecular_atmosphere=atmosphere_us_standard_ckd
        )
        si = default_spectral_index(atmosphere.molecular_atmosphere)
        kernel_context = KernelContext(si=si)

    else:
        component = ParticleEnsemble()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_ensembles=[component]
        )
        kernel_context = KernelContext()

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, ctx=kernel_context)


@pytest.mark.slow
@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_multi_mono(mode_mono, geometry, atmosphere_us_standard_mono):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=atmosphere_us_standard_mono,
        particle_ensembles=[ParticleEnsemble() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    si = default_spectral_index(atmosphere.molecular_atmosphere)
    kernel_context = KernelContext(si=si)
    check_scene_element(atmosphere, ctx=kernel_context)


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_multi_ckd(mode_ckd, geometry, atmosphere_us_standard_ckd):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    atmosphere = HeterogeneousAtmosphere(
        geometry={"type": geometry, "grid": np.linspace(0, 120, 121) * ureg.km},
        molecular_atmosphere=atmosphere_us_standard_ckd,
        particle_ensembles=[ParticleEnsemble() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    si = default_spectral_index(atmosphere.molecular_atmosphere)
    kernel_context = KernelContext(si=si)
    check_scene_element(atmosphere, ctx=kernel_context)


@pytest.mark.parametrize("field", ["sigma_a", "sigma_t"])
def test_heterogeneous_mix_collision_coefficients(modes_all_double, field):
    """
    Check for component mixing correctness. We expect that the absorption and
    extinction coefficients properly add up.
    """
    with ucc.override(length="km"):
        component_1 = ParticleEnsemble(bottom=0.0, top=1.25)
        component_2 = ParticleEnsemble(bottom=0.5, top=1.5)
        component_3 = ParticleEnsemble(bottom=0.75, top=2.0)

    mixed = HeterogeneousAtmosphere(
        geometry={
            "type": "plane_parallel",
            "grid": np.linspace(0, 120, 1201) * ureg.km,
        },
        particle_ensembles=[component_1, component_2, component_3],
    )
    ctx = KernelContext()
    grid = mixed.geometry.grid

    # Evaluate all profiles on the container's altitude grid
    radprofiles = {}

    for component, atmosphere in [
        ("component_1", component_1),
        ("component_2", component_2),
        ("component_3", component_3),
        ("mixed", mixed),
    ]:
        radprofiles[component] = atmosphere.eval_radprops(
            ctx.si, grid, optional_fields=True
        )

    collision_coefficient = {}
    for z in [0.1, 0.6, 1.0, 1.4, 1.9] * ureg.km:
        values = {}

        for component, radprofile in radprofiles.items():
            z_units = ureg.Unit(radprofile.coords["z_layer"].attrs["units"])

            field_units = ureg(radprofile.data_vars["sigma_a"].attrs["units"])
            val = (
                radprofile.data_vars[field]
                .interp(
                    z_layer=float(z.m_as(z_units)),
                    kwargs={"fill_value": 0.0},
                    method="nearest",
                )
                .squeeze()
            )
            values[component] = float(val) * field_units

        collision_coefficient[z.m] = values

    components = sorted(set(radprofiles.keys()) - {"mixed"})

    for z in collision_coefficient:
        total = collision_coefficient[z]["mixed"]
        expected = sum(collision_coefficient[z][component] for component in components)
        np.testing.assert_allclose(
            total.m_as(ureg.m**-1),
            expected.m_as(ureg.m**-1),
            err_msg=f"Failed for altitude {z = }",
        )


def test_heterogeneous_mix_weights(
    modes_all_double, atmosphere_us_standard_mono, atmosphere_us_standard_ckd
):
    """
    Check that component weights are correctly computed.
    """
    ctx = KernelContext()
    geometry = SceneGeometry.convert(
        {
            "type": "plane_parallel",
            "ground_altitude": 0.0 * ureg.km,
            "toa_altitude": 100.0 * ureg.km,
            "grid": np.linspace(0, 100, 101) * ureg.km,
        }
    )

    # Fist basic check: a uniform layer and a molecular atmosphere
    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=(
            atmosphere_us_standard_ckd
            if eradiate.mode().is_ckd
            else atmosphere_us_standard_mono
        ),
        particle_ensembles=ParticleEnsemble(
            bottom=0.0 * ureg.km,
            top=50.0 * ureg.km,
            distribution={"type": "uniform"},
        ),
    )
    template, _ = traverse(mixed.phase)
    mi_phase = mi.load_dict(template.render(ctx))
    mi_params = mi.traverse(mi_phase)

    # Weights should be non-zero over the first 50 km, and 0 above
    # (all to the molecular component)
    weights = np.squeeze(mi_params["weight_1.data"])
    assert len(weights) == geometry.grid.n_layers

    middle = np.argwhere(geometry.grid.layers <= 50.0 * ureg.km).max() + 1

    assert np.all((weights[:middle] > 0.0) & (weights[:middle] < 1.0))
    assert np.all(weights[middle:] == 0.0)

    # Second check: simple disjoint components, more than 1
    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        particle_ensembles=[
            ParticleEnsemble(
                bottom=0.0 * ureg.km,
                top=50.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleEnsemble(
                bottom=50.0 * ureg.km,
                top=80.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleEnsemble(
                bottom=80.0 * ureg.km,
                top=100.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
        ],
    )
    mi_wrapper = check_scene_element(
        mixed.phase.with_normalized_weights(ctx), mi.PhaseFunction
    )
    weight_0 = np.squeeze(mi_wrapper.parameters["weight_0.data"])
    weight_1 = np.squeeze(mi_wrapper.parameters["weight_1.data"])
    weight_2 = np.squeeze(mi_wrapper.parameters["weight_2.data"])

    middle = np.argwhere(geometry.grid.layers <= 50.0 * ureg.km).max() + 1
    fourfive = np.argwhere(geometry.grid.layers <= 80.0 * ureg.km).max() + 1

    assert np.all(weight_0[:middle] == 1.0)
    assert np.all(weight_0[middle:] == 0.0)
    assert np.all(weight_1[middle:fourfive] == 1.0)
    assert np.all(weight_1[:middle] == 0.0) and np.all(weight_1[fourfive:] == 0.0)
    assert np.all(weight_2[:fourfive] == 0.0)
    assert np.all(weight_2[fourfive:] == 1.0)

    # Third check: overlapping components
    # Component 1 has twice the optical thickness and extent of component 2,
    # therefore they have the same extinction coefficient
    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        particle_ensembles=[
            ParticleEnsemble(
                bottom=0.0 * ureg.km,
                top=100.0 * ureg.km,
                tau_ref=1.0,
                distribution={"type": "uniform"},
            ),
            ParticleEnsemble(
                bottom=50.0 * ureg.km,
                top=100.0 * ureg.km,
                tau_ref=0.5,
                distribution={"type": "uniform"},
            ),
        ],
    )
    mi_wrapper = check_scene_element(
        mixed.phase.with_normalized_weights(ctx), mi.PhaseFunction
    )
    weights_0 = np.squeeze(mi_wrapper.parameters["weight_0.data"])
    weights_1 = np.squeeze(mi_wrapper.parameters["weight_1.data"])
    middle = np.argwhere(geometry.grid.layers <= 50.0 * ureg.km).max() + 1

    assert np.all(weights_0[:middle] == 1.0)
    assert np.all(weights_0[middle:] == 0.5)
    assert np.all(weights_1[:middle] == 0.0)
    assert np.all(weights_1[middle:] == 0.5)


@pytest.mark.slow
def test_heterogeneous_scale(mode_mono, atmosphere_us_standard_mono):
    atmosphere = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=atmosphere_us_standard_mono,
        particle_ensembles=[ParticleEnsemble() for _ in range(2)],
        scale=2.0,
    )
    template, _ = traverse(atmosphere)
    assert template["medium_atmosphere.scale"] == 2.0

    # The scene element produces valid kernel dictionary specifications
    si = default_spectral_index(atmosphere.molecular_atmosphere)
    kernel_context = KernelContext(si=si)
    check_scene_element(atmosphere, ctx=kernel_context)


def test_heterogeneous_blend_switches(
    mode_mono,
    atmosphere_us_standard_mono,
):
    # Rayleigh-only atmosphere + particle ensemble combination works
    assert HeterogeneousAtmosphere(
        molecular_atmosphere=atmosphere_us_standard_mono,
        particle_ensembles=[ParticleEnsemble()],
    )


@pytest.mark.parametrize(
    "particle_radprops",
    ["particle_properties_absorbing_only", "particle_properties_scattering_only"],
)
def test_heterogeneous_absorbing_mol_atm(
    mode_ckd, particle_radprops, request, atmosphere_us_standard_ckd
):
    """
    Phase function weights are correct when the molecular atmosphere is
    absorbing-only and the particle ensemble is either absorbing-only or
    scattering-only.
    """
    # Expand fixture
    _particle_radprops = request.getfixturevalue(particle_radprops)
    # Create the heterogeneous atmosphere
    pe_bottom = 1.0 * ureg.km  # arbitrary
    pe_top = 4.0 * ureg.km  # arbitrary
    particle_ensemble = ParticleEnsemble(
        bottom=pe_bottom, top=pe_top, particle_properties=_particle_radprops
    )
    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=atmosphere_us_standard_ckd,
        particle_ensembles=particle_ensemble,
        geometry={
            "type": "spherical_shell",  # arbitrary
            "grid": np.linspace(0, 120, 121) * ureg.km,
        },
    )

    # Collect phase function weights
    mi_wrapper = check_scene_element(atmosphere.phase, mi.PhaseFunction)
    weights = np.squeeze(mi_wrapper.parameters["weight_1.volume.data"])

    # Extract phase function weights
    inside_particle_ensemble = (atmosphere.geometry.grid.layers >= pe_bottom) & (
        atmosphere.geometry.grid.layers <= pe_top
    )

    # Outside the particle ensemble, the phase function weight should be zero.
    assert np.all(weights.T[~inside_particle_ensemble] == 0.0)

    # Within the particle ensemble, the phase function weight should be:
    #   - zero, if the particle ensemble is not scattering (i.e., absorbing-only)
    #   - larger than zero, if the particle ensemble is scattering
    if particle_radprops == "particle_properties_absorbing_only":
        assert np.all(weights.T[inside_particle_ensemble] == 0.0)
    elif particle_radprops == "particle_properties_scattering_only":
        assert np.all(weights.T[inside_particle_ensemble] > 0.0)
    else:
        raise ValueError(
            f"Test parametrisation inconsistent. Expected 'absorbing_only' or "
            f"'scattering_only' (got {particle_radprops})"
        )


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("force_majorant", [False, True])
def test_heterogeneous_medium_type(
    mode_mono, geometry, force_majorant, atmosphere_us_standard_mono
):
    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=atmosphere_us_standard_mono,
        force_majorant=force_majorant,
    )
    template = atmosphere._template_medium

    if geometry == "plane_parallel":
        if force_majorant:
            assert template["type"] == "eoheterogeneous"
        else:
            assert template["type"] == "piecewise"
    elif geometry == "spherical_shell":
        assert template["type"] == "eoheterogeneous"


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("force_majorant", [False, True])
def test_heterogenous_extremum_type(
    mode_mono, geometry, force_majorant, atmosphere_us_standard_mono
):
    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=atmosphere_us_standard_mono,
        force_majorant=force_majorant,
        extremum_resolution=(1, 1, 1),
    )

    template = atmosphere._template_medium
    assert "extremum" not in template

    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=atmosphere_us_standard_mono,
        force_majorant=force_majorant,
        extremum_resolution=(1, 1, 12) if geometry == "plane_parallel" else (12, 1, 1),
    )
    template = atmosphere._template_medium

    if geometry == "plane_parallel":
        if force_majorant:
            assert "extremum" in template
            assert template["extremum"]["type"] == "extremum_grid"
        else:
            assert "extremum" not in template
    elif geometry == "spherical_shell":
        assert "extremum" in template
        assert template["extremum"]["type"] == "extremum_spherical"


@pytest.mark.parametrize(
    "wrap_mode", [WrapMode.CLAMP, WrapMode.REPEAT, WrapMode.MIRROR]
)
def test_heterogeneous_extremum_wrap_mode(
    mode_mono, wrap_mode, atmosphere_us_standard_mono
):
    """
    The extremum structure's wrap_mode always mirrors the geometry's
    wrap_mode, for each of the three supported modes.
    """
    atmosphere = HeterogeneousAtmosphere(
        geometry={"type": "plane_parallel", "wrap_mode": wrap_mode},
        molecular_atmosphere=atmosphere_us_standard_mono,
        force_majorant=True,
        extremum_resolution=(1, 1, 12),
    )
    template = atmosphere._template_medium
    assert template["extremum"]["wrap_mode"] == str(wrap_mode)


def test_heterogeneous_medium_aabb(mode_mono, atmosphere_us_standard_mono):
    """
    The medium AABB matches the geometry bbox horizontally; its vertical
    minimum is the grid's bottom level, not the shape bbox's own minimum.
    """
    atmosphere = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=atmosphere_us_standard_mono,
        force_majorant=True,
    )
    template = atmosphere._template_medium
    geometry = atmosphere.geometry

    assert "aabb_min" in template and "aabb_max" in template
    bbox_min = geometry.bbox.min.m_as("m")
    bbox_max = geometry.bbox.max.m_as("m")

    np.testing.assert_allclose(template["aabb_min"][:2], bbox_min[:2])
    np.testing.assert_allclose(template["aabb_max"], bbox_max)

    grid_bottom = geometry.grid.levels[0].m_as("m")
    assert template["aabb_min"][2] == pytest.approx(grid_bottom)
    assert template["aabb_min"][2] != pytest.approx(bbox_min[2])


def test_heterogeneous_medium_type_genuine_3d_grid(
    mode_mono, atmosphere_us_standard_mono
):
    """
    A non-onedim horizontal grid forces the eoheterogeneous medium type even
    with force_majorant left at its default (False).
    """
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        levels=np.linspace(0.0, 10.0, 6) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    assert not geometry.grid.onedim

    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=atmosphere_us_standard_mono,
    )
    template = atmosphere._template_medium

    assert template["type"] == "eoheterogeneous"
    assert "aabb_min" in template and "aabb_max" in template
