import numpy as np
import pytest
import xarray as xr

from eradiate import path_resolver
from eradiate.contexts import KernelDictContext
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleLayer,
)
from eradiate.scenes.atmosphere._heterogeneous import blend_radprops, overlapping
from eradiate.scenes.core import KernelDict
from eradiate.units import unit_registry as ureg


@pytest.fixture
def ussa76_approx_test_absorption_data_set():
    return path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_heterogeneous_ids():
    """
    Validator raises when two components have the same id.
    """
    with pytest.raises(ValueError):
        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere(id="qwerty"),
            particle_layers=[ParticleLayer(id="qwerty")],
        )

    with pytest.raises(ValueError):
        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=MolecularAtmosphere(id="azerty"),
            particle_layers=[ParticleLayer(id="qwerty"), ParticleLayer(id="qwerty")],
        )


def test_heterogeneous_kernel_phase_0(mode_mono):
    """
    If there is no molecular atmosphere and no particle layers, 'kernel_phase'
    returns an empty dictionary.
    """
    atmosphere = HeterogeneousAtmosphere()
    assert atmosphere.kernel_phase(ctx=KernelDictContext()) == {}


def test_heterogeneous_kernel_phase_1(mode_mono):
    """
    If there is a molecular atmosphere but no particle layers, 'kernel_phase'
    returns a dictionary that matches the dictionary returned by the molecular
    atmosphere's 'kernel_phase' method.
    """
    ctx = KernelDictContext()
    molecular_atmosphere = MolecularAtmosphere()
    atmosphere = HeterogeneousAtmosphere(molecular_atmosphere=molecular_atmosphere)
    assert atmosphere.kernel_phase(ctx=ctx) == molecular_atmosphere.kernel_phase(
        ctx=ctx
    )


def test_heterogeneous_kernel_phase_2(mode_mono):
    """
    If there is no molecular atmosphere but one particle layer, 'kernel_phase'
    returns a dictionary that matches the dictionary returned by the particle
    layer's 'kernel_phase' method.
    """
    ctx = KernelDictContext()
    particle_layer = ParticleLayer()
    atmosphere = HeterogeneousAtmosphere(particle_layers=[particle_layer])
    assert atmosphere.kernel_phase(ctx=ctx) == particle_layer.kernel_phase(ctx=ctx)


def test_heterogeneous_kernel_phase_3(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    """
    If there is a molecular atmosphere and one particle layer, 'kernel_phase'
    returns a 'blendphase' plugin dictionary with two nested phase function
    plugins directionaries matching the molecular atmosphere and particle
    layer phase function plugin dictionaries, respectively.
    """
    ctx = KernelDictContext()
    molecular_atmosphere = MolecularAtmosphere(
        id="molecules",
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set),
    )
    particle_layer = ParticleLayer(id="particles")
    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=molecular_atmosphere, particle_layers=[particle_layer]
    )
    phase = atmosphere.kernel_phase(ctx=ctx)
    assert phase.data["phase_atmosphere"]["type"] == "blendphase"
    assert (
        phase.data["phase_atmosphere"]["phase_molecules"]
        == molecular_atmosphere.kernel_phase(ctx=ctx).data["phase_molecules"]
    )
    assert (
        phase.data["phase_atmosphere"]["phase_particles"]
        == particle_layer.kernel_phase(ctx=ctx).data["phase_particles"]
    )


def test_heterogeneous_kernel_media_0(mode_mono):
    """
    If there is no molecular atmosphere and no particle layers, 'kernel_media'
    returns a 'heterogeneous' medium plugin specification dictionary.
    """
    atmosphere = HeterogeneousAtmosphere()
    medium = atmosphere.kernel_media(ctx=KernelDictContext())
    assert medium.data["medium_atmosphere"]["type"] == "heterogeneous"


def test_heterogeneous_kernel_media_1(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    """
    If there is a molecular atmosphere but no particle layers, 'kernel_media'
    returns a dictionary that matches the dictionary returned by the molecular
    atmosphere's 'kernel_media' method.
    """
    ctx = KernelDictContext()
    molecular_atmosphere = MolecularAtmosphere(
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set)
    )
    atmosphere = HeterogeneousAtmosphere(molecular_atmosphere=molecular_atmosphere)
    assert atmosphere.kernel_media(ctx=ctx) == molecular_atmosphere.kernel_media(
        ctx=ctx
    )


def test_heterogeneous_kernel_media_2(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    """
    If there is no molecular atmosphere but one particle layer, 'kernel_media'
    returns a dictionary that matches the dictionary returned by the particle
    layer's 'kernel_media' method.
    """
    ctx = KernelDictContext()
    particle_layer = ParticleLayer()
    atmosphere = HeterogeneousAtmosphere(particle_layers=[particle_layer])
    assert atmosphere.kernel_media(ctx=ctx) == particle_layer.kernel_media(ctx=ctx)


def test_heterogeneous_kernel_media_3(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    """
    If there is a molecular atmosphere and one particle layer, 'kernel_media'
    returns 'heterogeneous' plugin dictionary
    """
    ctx = KernelDictContext(ref=True)
    molecular_atmosphere = MolecularAtmosphere(
        id="molecules",
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set),
    )
    particle_layer = ParticleLayer(id="particles")
    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=molecular_atmosphere, particle_layers=[particle_layer]
    )
    media = atmosphere.kernel_media(ctx=ctx).data["medium_atmosphere"]
    assert media["type"] == "heterogeneous"
    assert media["phase"]["type"] == "ref"
    assert media["phase"]["id"] == "phase_atmosphere"


@pytest.fixture
def test_molecules_radprops():
    """
    Test molecules radiative property data set.
    """
    z_level = np.linspace(0, 10, 11)
    z_layer = (z_level[:-1] + z_level[1:]) / 2.0
    return xr.Dataset(
        data_vars={
            "albedo": ("z_layer", np.ones(10), dict(units="dimensionless")),
            "sigma_t": ("z_layer", 1.0 + np.random.random(10), dict(units="km^-1")),
        },
        coords={
            "z_layer": ("z_layer", z_layer, dict(units="km")),
            "z_level": ("z_level", z_level, dict(units="km")),
        },
    )


@pytest.fixture
def test_particles_radprops():
    """
    Test particles radiative property data set.
    """
    z_level = np.linspace(0, 10, 11)
    z_layer = (z_level[:-1] + z_level[1:]) / 2.0
    return xr.Dataset(
        data_vars={
            "albedo": ("z_layer", np.ones(10), dict(units="dimensionless")),
            "sigma_t": ("z_layer", 1.0 + np.random.random(10), dict(units="km^-1")),
        },
        coords={
            "z_layer": ("z_layer", z_layer, dict(units="km")),
            "z_level": ("z_level", z_level, dict(units="km")),
        },
    )


def test_blend_radprops_sigma_t(
    mode_mono, test_molecules_radprops, test_particles_radprops
):
    """
    Returned radiative property data set has a 'sigma_t' that matches
    the sum of 'sigma_t's of the molecules and particles' radiative
    property data sets.
    """
    molecules = test_molecules_radprops
    particles = test_particles_radprops
    blended, _ = blend_radprops(background=molecules, foreground=particles)

    assert np.allclose(
        blended.sigma_t.values,
        molecules.sigma_t.values + particles.sigma_t.values,
    )


def test_blend_radprops_albedo(
    mode_mono, test_molecules_radprops, test_particles_radprops
):
    """
    Returned radiative property data set has an 'albedo' that matches
    the weighted sum of 'albedo's of the molecules and particles' radiative
    property data sets, where the weights are given by the ratio of the
    molecules and particles extinction coefficient to the total extinction
    coefficient.
    """
    molecules = test_molecules_radprops
    particles = test_particles_radprops
    blended, _ = blend_radprops(background=molecules, foreground=molecules)
    sigma_t_molecules = molecules.sigma_t
    sigma_t_particles = particles.sigma_t
    sigma_t_total = sigma_t_molecules + sigma_t_particles
    albedo_molecules = molecules.albedo
    albedo_particles = particles.albedo
    expected_albedo = (sigma_t_molecules / sigma_t_total) * albedo_molecules + (
        sigma_t_particles / sigma_t_total
    ) * albedo_particles

    assert np.allclose(
        blended.albedo.values,
        expected_albedo.values,
    )


def test_blend_radprops_ratio(
    mode_mono, test_molecules_radprops, test_particles_radprops
):
    """
    Returned blending ratio matches the ratio of the particles' extinction
    coefficient with the molecules's extinction coefficient.
    """
    molecules = test_molecules_radprops
    particles = test_particles_radprops
    _, ratio = blend_radprops(background=molecules, foreground=particles)
    expected_ratio = particles.sigma_t / (particles.sigma_t + molecules.sigma_t)

    assert np.allclose(
        ratio.values,
        expected_ratio.values,
    )


def test_blend_radprops_zeros_radprops(
    mode_mono, test_molecules_radprops, test_particles_radprops
):
    """
    Returns a data set filled with zeros when the molecules and particles radiative
    property data sets both have a zero extinction coefficient.
    """
    z_level = np.linspace(0, 10, 11)
    z_layer = (z_level[:-1] + z_level[1:]) / 2.0
    molecules = xr.Dataset(
        data_vars={
            "albedo": ("z_layer", np.zeros(10), dict(units="dimensionless")),
            "sigma_t": ("z_layer", np.zeros(10), dict(units="km^-1")),
        },
        coords={
            "z_layer": ("z_layer", z_layer, dict(units="km")),
            "z_level": ("z_level", z_level, dict(units="km")),
        },
    )

    particles = xr.Dataset(
        data_vars={
            "albedo": ("z_layer", np.zeros(10), dict(units="dimensionless")),
            "sigma_t": ("z_layer", np.zeros(10), dict(units="km^-1")),
        },
        coords={
            "z_layer": ("z_layer", z_layer, dict(units="km")),
            "z_level": ("z_level", z_level, dict(units="km")),
        },
    )

    blended, ratio = blend_radprops(background=molecules, foreground=particles)

    assert np.allclose(blended.sigma_t.values, 0.0)
    assert np.allclose(blended.albedo, 0.0)
    assert np.allclose(ratio, 0.5)


def test_interpolate_radprops():
    pass


@pytest.fixture
def test_particle_layers():
    """
    Fixture producing ten particle layers from 0 to 10 km (1km thick each).
    """
    return [
        ParticleLayer(id=f"layer_{int(z)}", bottom=z * ureg.km, top=(z + 1.0) * ureg.km)
        for z in np.linspace(0.0, 10.0, 11)
    ]


def test_overlapping_0():
    """
    Returns empty list if 'particle_layers' is an empty list.
    """
    particle_layer = ParticleLayer(
        id="my_layer", bottom=0.0 * ureg.km, top=1.0 * ureg.km
    )
    assert overlapping(particle_layers=[], particle_layer=particle_layer) == []


def test_overlapping_1(test_particle_layers):
    """
    A layer at [0.5, 1.5] km overlaps the layer at [0.0, 1.0] km and the
    layer at [1.0, 2.0].
    """
    particle_layer = ParticleLayer(
        id="my_layer", bottom=0.5 * ureg.km, top=1.5 * ureg.km
    )
    assert (
        overlapping(particle_layers=test_particle_layers, particle_layer=particle_layer)
        == test_particle_layers[:2]
    )


def test_overlapping_2(test_particle_layers):
    """
    A layer at [15.0, 16.0] km does not overlap layers below 10 km.
    """
    particle_layer = ParticleLayer(
        id="my_layer", bottom=15.0 * ureg.km, top=16.0 * ureg.km
    )
    assert (
        overlapping(particle_layers=test_particle_layers, particle_layer=particle_layer)
        == []
    )


def test_overlapping_3(test_particle_layers):
    """
    A layer at [0.0, 1.0] km overlaps the layer at [0.0, 1.0] km.
    """
    particle_layer = ParticleLayer(
        id="my_layer", bottom=0.0 * ureg.km, top=1.0 * ureg.km
    )
    assert (
        overlapping(particle_layers=test_particle_layers, particle_layer=particle_layer)
        == test_particle_layers[0:1]
    )
