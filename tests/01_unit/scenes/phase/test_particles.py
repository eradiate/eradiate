import mitsuba as mi
import pytest

import eradiate
from eradiate import fresolver
from eradiate.radprops import ParticleProperties
from eradiate.scenes.phase import ParticlePhaseFunction
from eradiate.spectral import SpectralIndex
from eradiate.test_tools.types import check_scene_element

DS_ID_TO_FNAME = {
    "unpolarized_data": "govaerts_2021-desert-aer_core_v2",
    "polarized_data": "aeronet_sahara_spherical_RAMIA_GENERIC_extrapolated-aer_core_v2",
}


@pytest.fixture(scope="module", params=list(DS_ID_TO_FNAME.keys()))
def pds(request):
    ds_id = request.param
    fname = DS_ID_TO_FNAME[ds_id]
    yield fresolver.load_dataset(f"tests/aerosol/{fname}.nc")


@pytest.fixture(scope="module")
def pprops(pds):
    yield ParticleProperties(pds)


@pytest.fixture(scope="module")
def pphase(pprops):
    yield ParticlePhaseFunction(particle_properties=pprops)


class TestParticlePhaseFunction:
    def test_basics(self, modes_all_double, pprops):
        # Object can be constructed in all modes
        ppf = ParticlePhaseFunction(particle_properties=pprops)

        # If object has polarized data, it activates polarization in polarized
        # modes
        assert ppf.is_polarized == (
            ppf.particle_properties.has_polarization
            and eradiate.get_mode().is_polarized
        )

        check_scene_element(ppf, mi.PhaseFunction)

    def test_eval_impl(self, modes_all_double, pphase):
        # Repeated evaluations with the same wavelength value do not trigger a
        # recomputation
        si = SpectralIndex.new()
        a = pphase._eval_impl(si.w)
        b = pphase._eval_impl(si.w)
        assert a is b

    def test_eval_mu(self, modes_all_double, pphase):
        si = SpectralIndex.new()
        result = pphase.eval_mu(si)
        assert result.size == pphase.particle_properties.data.sizes["iangle"]

        # Repeated evaluations with the same wavelength value do not trigger a
        # recomputation
        assert pphase.eval_mu(si) is result

    def test_eval_phase(self, modes_all_double, pphase):
        si = SpectralIndex.new()
        result = pphase.eval_phase(si)
        assert result.shape == (
            pphase.particle_properties.data.sizes["phamat"],
            pphase.particle_properties.data.sizes["iangle"],
        )

        # Repeated evaluations with the same wavelength value do not trigger a
        # recomputation
        assert pphase.eval_phase(si) is result
