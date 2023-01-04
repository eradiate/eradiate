import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import CKDSpectralContext, KernelDictContext
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleLayer,
)
from eradiate.scenes.core import traverse
from eradiate.test_tools.types import check_scene_element


@pytest.fixture
def path_to_ussa76_approx_data():
    return eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_heterogeneous_empty(modes_all_double):
    # Passing no component is not allowed
    with pytest.raises(ValueError):
        HeterogeneousAtmosphere()


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("component", ["molecular", "particle"])
def test_heterogeneous_single_mono(
    mode_mono, geometry, component, path_to_ussa76_approx_data
):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if component == "molecular":
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry,
            molecular_atmosphere=MolecularAtmosphere.ussa_1976(
                absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
            ),
        )

    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_layers=[component]
        )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelDictContext())


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
            geometry=geometry, molecular_atmosphere=MolecularAtmosphere.afgl_1986()
        )

    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_layers=[component]
        )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelDictContext())


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
def test_heterogeneous_multi_mono(mode_mono, geometry, path_to_ussa76_approx_data):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    molecular_atmosphere = MolecularAtmosphere.ussa_1976(
        absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
    )

    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=molecular_atmosphere,
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelDictContext())


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_heterogeneous_multi_ckd(mode_ckd, geometry, bin_set):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    molecular_atmosphere = MolecularAtmosphere.afgl_1986()

    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=molecular_atmosphere,
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # The scene element produces valid kernel dictionary specifications
    check_scene_element(atmosphere, KernelDictContext())


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
        geometry="plane_parallel",
        particle_layers=[component_1, component_2, component_3],
    )
    ctx = KernelDictContext()

    radprofiles = {
        component: atmosphere.eval_radprops(ctx.spectral_ctx, optional_fields=True)
        for component, atmosphere in [
            ("component_1", component_1),
            ("component_2", component_2),
            ("component_3", component_3),
            ("mixed", mixed),
        ]
    }

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
        assert np.allclose(total, expected), f"{z = }"


def test_heterogeneous_mix_weights(modes_all_double):
    """
    Check that component weights are correctly computed.
    """
    ctx = KernelDictContext()

    # Fist basic check: a uniform layer and a molecular atmosphere
    molecular = (
        MolecularAtmosphere.afgl_1986(levels=np.linspace(0, 100, 101) * ureg.km)
        if eradiate.mode().is_ckd
        else MolecularAtmosphere.ussa_1976(levels=np.linspace(0, 100, 101) * ureg.km)
    )

    mixed = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=molecular,
        particle_layers=ParticleLayer(
            bottom=0.0 * ureg.km,
            top=50.0 * ureg.km,
            distribution={"type": "uniform"},
        ),
    )
    print(f"{mixed.phase.eval_conditional_weights(ctx.spectral_ctx) = }")  # OK
    template, params = traverse(mixed.phase)
    mi_phase = mi.load_dict(template.render(ctx))
    mi_params = mi.traverse(mi_phase)
    mi_params["weight.data"] = np.array(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32
    ).reshape((-1, 1, 1, 1))
    mi_params.update()
    print(np.array(mi_params["weight.data"]))
    assert False
    # We have a bug here: the first two weights are set to nonsense values
    mi_params.update(params.render(ctx))
    print(np.array(mi_params["weight.data"]))
    mi_phase, mi_params = check_scene_element(mixed.phase, mi.PhaseFunction)

    # Weights should be non zero over the first 50 km, and 0 above
    # (all to the molecular component)
    print(np.array(mi_params["weight.data"]))
    weights = np.squeeze(mi_params["weight.data"])
    assert len(weights) == len(mixed.zgrid)
    middle = len(weights) // 2

    print(f"{mixed._eval_sigma_t_impl(ctx.spectral_ctx) = }")
    print(f"{mixed._eval_sigma_s_impl(ctx.spectral_ctx) = }")
    print(f"{weights = }")

    assert np.all((weights[:middle] > 0.0) & (weights[:middle] < 1.0))
    assert np.all(weights[middle:] == 0.0)

    # Second check: simple disjoint components, more than 1
    mixed = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        particle_layers=[
            ParticleLayer(
                bottom=0.0 * ureg.km,
                top=50.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleLayer(
                bottom=50.0 * ureg.km,
                top=75.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
            ParticleLayer(
                bottom=75.0 * ureg.km,
                top=100.0 * ureg.km,
                distribution={"type": "uniform"},
            ),
        ],
    )
    mi_phase, mi_params = check_scene_element(mixed.phase, mi.PhaseFunction)
    weight_1 = np.squeeze(mi_params["weight.data"])
    weight_2 = np.squeeze(mi_params["phase_1.weight.data"])
    middle = len(weight_1) // 2
    threeq = len(weight_1) * 3 // 4

    assert np.all(weight_1[:middle] == 0.0)
    assert np.all(weight_1[middle:] == 1.0)
    assert np.all(weight_2[:threeq] == 0.0)
    assert np.all(weight_2[threeq:] == 1.0)

    # Third check: overlapping components
    # Component 1 has twice the optical thickness and extent of component 2,
    # therefore they have the same extinction coefficient
    mixed = HeterogeneousAtmosphere(
        geometry="plane_parallel",
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
    phase_mixed = mixed.kernel_phase(ctx).load()
    params = mi.traverse(phase_mixed)
    weights = np.squeeze(params["weight.data"])
    middle = len(weights) // 2

    assert np.all(weights[:middle] == 0.0)
    assert np.all(weights[middle:] == 0.5)


def test_heterogeneous_scale(mode_mono, path_to_ussa76_approx_data):
    ctx = KernelDictContext()
    d = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=MolecularAtmosphere.ussa_1976(
            absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
        ),
        particle_layers=[ParticleLayer() for _ in range(2)],
        scale=2.0,
    ).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert isinstance(d.load(), mi.Scene)


def test_heterogeneous_blend_switches(mode_mono):
    # Rayleigh-only atmosphere + particle layer combination works
    assert HeterogeneousAtmosphere(
        molecular_atmosphere=MolecularAtmosphere.ussa_1976(
            has_absorption=False, has_scattering=True
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
        geometry={"type": "spherical_shell"},  # arbitrary
    )

    # Generate kernel dict
    kernel_dict = atmosphere.kernel_phase(
        KernelDictContext(
            spectral_ctx=CKDSpectralContext(bin_set="1nm"),
        )
    )

    # Extract phase function weights
    weight = np.array(kernel_dict.data["phase"]["weight"]["grid"])
    z_mesh = atmosphere._high_res_z_layer().m_as(uck["length"])
    inside_particle_layer = (z_mesh >= pl_bottom.m_as(uck["length"])) & (
        z_mesh <= pl_top.m_as(uck["length"])
    )

    # Outside the particle layer, the phase function weight should be zero.
    assert np.all(weight[~inside_particle_layer] == 0.0)

    # Within the particle layer, the phase function weight should be:
    #   - zero, if the particle layer is not scattering (i.e., absorbing-only)
    #   - larger than zero, if the particle layer is scattering
    if particle_radprops == "absorbing_only":
        assert np.all(weight[inside_particle_layer] == 0.0)
    elif particle_radprops == "scattering_only":
        assert np.all(weight[inside_particle_layer] > 0.0)
    else:
        raise ValueError(
            f"Test parametrisation inconsistent. Expected 'absorbing_only' or "
            f"'scattering_only' (got {particle_radprops})"
        )
