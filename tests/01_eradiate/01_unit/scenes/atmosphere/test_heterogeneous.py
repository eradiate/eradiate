import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import KernelContext
from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.radprops import ZGrid
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleLayer,
)
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import SceneGeometry
from eradiate.test_tools.types import check_scene_element
from eradiate.test_tools.util import skipif_data_not_found


def test_heterogeneous_empty(modes_all_double):
    # Passing no component is not allowed
    with pytest.raises(ValueError):
        HeterogeneousAtmosphere()


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("component", ["molecular", "particle"])
def test_heterogeneous_single_mono(mode_mono, geometry, component):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if component == "molecular":
        skipif_data_not_found(
            f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
        )
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry,
            molecular_atmosphere=MolecularAtmosphere.ussa_1976(),
        )

    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_layers=[component]
        )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelContext())


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("component", ["molecular", "particle"])
@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_heterogeneous_single_ckd(mode_ckd, geometry, component, bin_set):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if component == "molecular":
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry,
            molecular_atmosphere=MolecularAtmosphere.afgl_1986(),
        )

    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry,
            particle_layers=[component],
        )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelContext())


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_multi_mono(mode_mono, geometry):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=MolecularAtmosphere.ussa_1976(),
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelContext())


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_heterogeneous_multi_ckd(mode_ckd, geometry, bin_set):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    atmosphere = HeterogeneousAtmosphere(
        geometry={"type": geometry, "zgrid": np.linspace(0, 120, 121) * ureg.km},
        molecular_atmosphere=MolecularAtmosphere.afgl_1986(),
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelContext())


@pytest.mark.parametrize("field", ["sigma_a", "sigma_t"])
def test_heterogeneous_mix_collision_coefficients(modes_all_double, field):
    """
    Check for component mixing correctness. We expect that the absorption and
    extinction coefficients properly add up.
    """
    with ucc.override(length="km"):
        component_1 = ParticleLayer(bottom=0.0, top=1.25)
        component_2 = ParticleLayer(bottom=0.5, top=1.5)
        component_3 = ParticleLayer(bottom=0.75, top=2.0)

    mixed = HeterogeneousAtmosphere(
        geometry={
            "type": "plane_parallel",
            "zgrid": np.linspace(0, 120, 1201) * ureg.km,
        },
        particle_layers=[component_1, component_2, component_3],
    )
    ctx = KernelContext()
    zgrid = mixed.geometry.zgrid

    # Evaluate all profiles on the container's altitude grid
    radprofiles = {}

    for component, atmosphere in [
        ("component_1", component_1),
        ("component_2", component_2),
        ("component_3", component_3),
        ("mixed", mixed),
    ]:
        radprofiles[component] = atmosphere.eval_radprops(
            ctx.si, zgrid, optional_fields=True
        )

    collision_coefficient = {}
    for z in [0.1, 0.6, 1.0, 1.4, 1.9] * ureg.km:
        values = {}

        for component, radprofile in radprofiles.items():
            z_units = ureg.Unit(radprofile.coords["z_layer"].attrs["units"])

            field_units = ureg(radprofile.data_vars["sigma_a"].attrs["units"])
            values[component] = (
                float(
                    radprofile.data_vars[field].interp(
                        z_layer=z.m_as(z_units),
                        kwargs={"fill_value": 0.0},
                        method="nearest",
                    )
                )
                * field_units
            )

        collision_coefficient[z.m] = values

    components = sorted(set(radprofiles.keys()) - {"mixed"})

    for z in collision_coefficient.keys():
        total = collision_coefficient[z]["mixed"]
        expected = sum(collision_coefficient[z][component] for component in components)
        np.testing.assert_allclose(
            total.m_as(ureg.m**-1),
            expected.m_as(ureg.m**-1),
            err_msg=f"Failed for altitude {z = }",
        )


def test_heterogeneous_mix_weights(modes_all_double):
    """
    Check that component weights are correctly computed.
    """
    ctx = KernelContext()
    geometry = SceneGeometry.convert(
        {
            "type": "plane_parallel",
            "ground_altitude": 0.0 * ureg.km,
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": ZGrid(np.linspace(0, 100, 101) * ureg.km),
        }
    )

    # Fist basic check: a uniform layer and a molecular atmosphere
    if eradiate.mode().is_mono:
        skipif_data_not_found(
            f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
        )
        molecular = MolecularAtmosphere.ussa_1976(levels=geometry.zgrid.levels)
    elif eradiate.mode().is_ckd:
        molecular = MolecularAtmosphere.afgl_1986(levels=geometry.zgrid.levels)
    else:
        raise NotImplementedError

    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=molecular,
        particle_layers=ParticleLayer(
            bottom=0.0 * ureg.km,
            top=50.0 * ureg.km,
            distribution={"type": "uniform"},
        ),
    )
    template, params = traverse(mixed.phase)
    mi_phase = mi.load_dict(template.render(ctx))
    mi_params = mi.traverse(mi_phase)

    # Weights should be non-zero over the first 50 km, and 0 above
    # (all to the molecular component)
    weights = np.squeeze(mi_params["weight.data"])
    assert len(weights) == geometry.zgrid.n_layers

    middle = np.argwhere(geometry.zgrid.layers <= 50.0 * ureg.km).max() + 1

    assert np.all((weights[:middle] > 0.0) & (weights[:middle] < 1.0))
    assert np.all(weights[middle:] == 0.0)

    # Second check: simple disjoint components, more than 1
    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        particle_layers=[
            ParticleLayer(
                bottom=0.0 * ureg.km,
                top=50.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleLayer(
                bottom=50.0 * ureg.km,
                top=80.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleLayer(
                bottom=80.0 * ureg.km,
                top=100.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
        ],
    )
    mi_wrapper = check_scene_element(mixed.phase, mi.PhaseFunction)
    weight_1 = np.squeeze(mi_wrapper.parameters["weight.data"])
    weight_2 = np.squeeze(mi_wrapper.parameters["phase_1.weight.data"])

    middle = np.argwhere(geometry.zgrid.layers <= 50.0 * ureg.km).max() + 1
    fourfive = np.argwhere(geometry.zgrid.layers <= 80.0 * ureg.km).max() + 1

    assert np.all(weight_1[:middle] == 0.0)
    assert np.all(weight_1[middle:] == 1.0)
    assert np.all(weight_2[:fourfive] == 0.0)
    assert np.all(weight_2[fourfive:] == 1.0)

    # Third check: overlapping components
    # Component 1 has twice the optical thickness and extent of component 2,
    # therefore they have the same extinction coefficient
    mixed = HeterogeneousAtmosphere(
        geometry=geometry,
        particle_layers=[
            ParticleLayer(
                bottom=0.0 * ureg.km,
                top=100.0 * ureg.km,
                tau_ref=1.0,
                distribution={"type": "uniform"},
            ),
            ParticleLayer(
                bottom=50.0 * ureg.km,
                top=100.0 * ureg.km,
                tau_ref=0.5,
                distribution={"type": "uniform"},
            ),
        ],
    )
    mi_wrapper = check_scene_element(mixed.phase, mi.PhaseFunction)
    weights = np.squeeze(mi_wrapper.parameters["weight.data"])
    middle = np.argwhere(geometry.zgrid.layers <= 50.0 * ureg.km).max() + 1

    assert np.all(weights[:middle] == 0.0)
    assert np.all(weights[middle:] == 0.5)


def test_heterogeneous_scale(mode_mono):
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    atmosphere = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=MolecularAtmosphere.ussa_1976(),
        particle_layers=[ParticleLayer() for _ in range(2)],
        scale=2.0,
    )
    template, params = traverse(atmosphere)
    assert template["medium_atmosphere.scale"] == 2.0

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelContext())


def test_heterogeneous_blend_switches(mode_mono):
    # Rayleigh-only atmosphere + particle layer combination works
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    assert HeterogeneousAtmosphere(
        molecular_atmosphere=MolecularAtmosphere.ussa_1976(
            has_absorption=False,
            has_scattering=True,
        ),
        particle_layers=[ParticleLayer()],
    )


@pytest.mark.parametrize(
    "particle_radprops",
    ["absorbing_only", "scattering_only"],
)
def test_heterogeneous_absorbing_mol_atm(mode_ckd, particle_radprops, request):
    """
    Phase function weights are correct when the molecular atmosphere is
    absorbing-only and the particle layer is either absorbing-only or
    scattering-only.
    """
    # Create the heterogeneous atmosphere
    pl_bottom = 1.0 * ureg.km  # arbitrary
    pl_top = 4.0 * ureg.km  # arbitrary
    particle_layer = ParticleLayer(
        bottom=pl_bottom,
        top=pl_top,
        dataset=request.getfixturevalue(particle_radprops),
    )
    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=MolecularAtmosphere.afgl_1986(
            has_absorption=True,
            has_scattering=False,
        ),
        particle_layers=particle_layer,
        geometry={
            "type": "spherical_shell",  # arbitrary
            "zgrid": np.linspace(0, 120, 121) * ureg.km,
        },
    )

    # Collect phase function weights
    mi_wrapper = check_scene_element(atmosphere.phase, mi.PhaseFunction)
    weights = np.squeeze(mi_wrapper.parameters["weight.volume.data"])

    # Extract phase function weights
    inside_particle_layer = (atmosphere.geometry.zgrid.layers >= pl_bottom) & (
        atmosphere.geometry.zgrid.layers <= pl_top
    )

    # Outside the particle layer, the phase function weight should be zero.
    assert np.all(weights[~inside_particle_layer] == 0.0)

    # Within the particle layer, the phase function weight should be:
    #   - zero, if the particle layer is not scattering (i.e., absorbing-only)
    #   - larger than zero, if the particle layer is scattering
    if particle_radprops == "absorbing_only":
        assert np.all(weights[inside_particle_layer] == 0.0)
    elif particle_radprops == "scattering_only":
        assert np.all(weights[inside_particle_layer] > 0.0)
    else:
        raise ValueError(
            f"Test parametrisation inconsistent. Expected 'absorbing_only' or "
            f"'scattering_only' (got {particle_radprops})"
        )
