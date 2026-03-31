import pytest

from eradiate import fresolver
from eradiate.radprops import ParticleProperties
from eradiate.scenes.phase import ParticlePhaseFunction
from eradiate.spectral import SpectralIndex


@pytest.fixture
def particle_dataset():
    yield fresolver.load_dataset("tests/aerosol/govaerts_2021-desert-aer_core_v2.nc")


@pytest.fixture
def particle_properties(particle_dataset):
    yield ParticleProperties(particle_dataset)


@pytest.fixture
def particle_phase_function(particle_properties):
    yield ParticlePhaseFunction(particle_properties=particle_properties)


class TestParticlePhaseFunction:
    def test_constructor(self, particle_properties):
        ParticlePhaseFunction(particle_properties=particle_properties)

    def test_eval(self, modes_all_double, particle_phase_function):
        si = SpectralIndex.new()
        print(particle_phase_function.eval(si))
