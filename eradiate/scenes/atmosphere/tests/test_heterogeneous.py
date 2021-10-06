import pytest

import eradiate
from eradiate import path_resolver
from eradiate._mode import ModeFlags
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


def test_atmosphere_heterogeneous_empty(modes_all_single):
    """
    Unit tests for a HeterogeneousAtmosphere with no components.
    """
    # Constructing without argument succeeds
    atmosphere = HeterogeneousAtmosphere()

    # Produced kernel dicts are all empty
    ctx = KernelDictContext()
    assert atmosphere.kernel_shapes(ctx).data == {}
    assert atmosphere.kernel_media(ctx).data == {}
    assert atmosphere.kernel_phase(ctx).data == {}
    assert atmosphere.kernel_dict(ctx).data == {}


@pytest.mark.parametrize("components", ("molecular", "particle"))
def test_atmosphere_heterogeneous_single(
    modes_all_single, components, path_to_ussa76_approx_data
):
    """
    Unit tests for a HeterogeneousAtmosphere with a single component.
    """
    # Construct succeeds
    if components == "molecular":
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            component = MolecularAtmosphere.ussa1976(
                absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
            )
        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            component = MolecularAtmosphere.afgl1986()
        else:
            pytest.skip(f"unsupported mode '{eradiate.mode().id}'")

        atmosphere = HeterogeneousAtmosphere(molecular_atmosphere=component)
    else:
        component = ParticleLayer()
        atmosphere = HeterogeneousAtmosphere(particle_layers=[component])

    # Produced kernel dict can be loaded
    ctx = KernelDictContext()
    assert atmosphere.kernel_dict(ctx).load()


def test_atmosphere_heterogeneous_multi(modes_all_single, path_to_ussa76_approx_data):
    """
    Unit tests for a HeterogeneousAtmosphere with multiple (2+) components.
    """
    # Construct succeeds
    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        molecular_atmosphere = MolecularAtmosphere.ussa1976(
            absorption_data_sets={"us76_u86_4": path_to_ussa76_approx_data},
        )
    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        molecular_atmosphere = MolecularAtmosphere.afgl1986()
    else:
        pytest.skip(f"unsupported mode '{eradiate.mode().id}'")

    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=molecular_atmosphere,
        particle_layers=[ParticleLayer() for _ in range(2)],
    )

    # Radiative property metadata are correct
    ctx = KernelDictContext()

    # Kernel dict production succeeds
    assert atmosphere.kernel_phase(ctx)
    assert atmosphere.kernel_media(ctx)
    assert atmosphere.kernel_shapes(ctx)

    # Produced kernel dict can be loaded
    kernel_dict = atmosphere.kernel_dict(ctx)
    assert kernel_dict.load()
