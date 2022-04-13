import mitsuba as mi
import numpy as np
import pytest

from eradiate import path_resolver
from eradiate.contexts import KernelDictContext
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleLayer,
)


@pytest.fixture
def path_to_ussa76_approx_data():
    return path_resolver.resolve(
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

    # Produced kernel dict can be loaded
    ctx = KernelDictContext()
    assert atmosphere.kernel_dict(ctx).load()


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

    # Produced kernel dict can be loaded
    ctx = KernelDictContext(spectral_ctx={"bin_set": bin_set})
    assert atmosphere.kernel_dict(ctx).load()


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

    ctx = KernelDictContext()

    # Kernel dict production succeeds
    assert atmosphere.kernel_phase(ctx)
    assert atmosphere.kernel_media(ctx)
    assert atmosphere.kernel_shapes(ctx)

    # Produced kernel dict can be loaded
    kernel_dict = atmosphere.kernel_dict(ctx)
    assert kernel_dict.load()


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

    ctx = KernelDictContext(spectral_ctx={"bin_set": bin_set})

    # Kernel dict production succeeds and produced data can be loaded
    assert isinstance(atmosphere.kernel_phase(ctx).load(), mi.PhaseFunction)
    assert isinstance(atmosphere.kernel_media(ctx).load(), mi.Medium)
    assert isinstance(atmosphere.kernel_shapes(ctx).load(), mi.Shape)

    kernel_dict = atmosphere.kernel_dict(ctx)
    assert isinstance(kernel_dict.load(), mi.Scene)


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
        component: atmosphere.eval_radprops(ctx.spectral_ctx)
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
        assert np.allclose(expected, total), f"{z = }"


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

    # Purely absorbing atmosphere + particle layer combination is not allowed
    with pytest.raises(ValueError):
        HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere.ussa_1976(
                has_absorption=True, has_scattering=False
            ),
            particle_layers=[ParticleLayer()],
        )
