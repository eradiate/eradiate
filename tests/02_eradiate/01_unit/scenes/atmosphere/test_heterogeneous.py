import pytest

import eradiate
from eradiate import path_resolver
from eradiate.contexts import CKDSpectralContext, KernelDictContext
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
@pytest.mark.parametrize("components", ["molecular", "particle"])
@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_heterogeneous_single(
    modes_all_double, geometry, components, bin_set, path_to_ussa76_approx_data
):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if components == "molecular":
        if eradiate.mode().is_mono:
            component = MolecularAtmosphere.ussa_1976(
                absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
            )
        elif eradiate.mode().is_ckd:
            component = MolecularAtmosphere.afgl_1986()
        else:
            pytest.skip(f"unsupported mode '{eradiate.mode().id}'")

        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, molecular_atmosphere=component
        )
    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(
            geometry=geometry, particle_layers=[component]
        )

    # Produced kernel dict can be loaded
    ctx = KernelDictContext(spectral_ctx=CKDSpectralContext(bin_set=bin_set))
    assert atmosphere.kernel_dict(ctx).load()


@pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
@pytest.mark.parametrize("bin_set", ["1nm", "10nm"])
def test_heterogeneous_multi(
    modes_all_double, geometry, bin_set, path_to_ussa76_approx_data
):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    if eradiate.mode().is_mono:
        molecular_atmosphere = MolecularAtmosphere.ussa_1976(
            absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
        )
    elif eradiate.mode().is_ckd:
        molecular_atmosphere = MolecularAtmosphere.afgl_1986()
    else:
        pytest.skip(f"unsupported mode '{eradiate.mode().id}'")

    atmosphere = HeterogeneousAtmosphere(
        geometry=geometry,
        molecular_atmosphere=molecular_atmosphere,
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # Radiative property metadata are correct
    ctx = KernelDictContext(spectral_ctx=CKDSpectralContext(bin_set=bin_set))

    # Kernel dict production succeeds
    assert atmosphere.kernel_phase(ctx)
    assert atmosphere.kernel_media(ctx)
    assert atmosphere.kernel_shapes(ctx)

    # Produced kernel dict can be loaded
    kernel_dict = atmosphere.kernel_dict(ctx)
    assert kernel_dict.load()


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
    assert d.load()


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
