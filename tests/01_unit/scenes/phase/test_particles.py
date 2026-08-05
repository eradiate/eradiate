import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

import eradiate
from eradiate import fresolver
from eradiate.data.convert import make_aer_core_v2
from eradiate.radprops import ParticleProperties
from eradiate.scenes.phase import ParticlePhaseFunction
from eradiate.spectral import SpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import unit_registry as ureg

DS_ID_TO_FNAME = {
    "unpolarized_data": "govaerts_2021-desert-aer_core_v2",
    "polarized_data": "aeronet_sahara_spherical_RAMIA_GENERIC_extrapolated-aer_core_v2",
    "pmom_data": "soot.mie-aer_core_v2",
}


def make_minimal_aer_core_v2(nphamat: int) -> xr.Dataset:
    """
    Build a minimal Aer-Core v2 dataset with ``nphamat`` phase matrix components
    (1: unpolarized, 4: spherical, 6: spheroidal).
    """
    # Phase matrix component names, in Aer-Core v2 storage order
    phamat_names = ["11", "12", "33", "34", "22", "44"]

    w = np.array([400.0, 700.0])
    mu = np.tile(np.array([-1.0, 0.0, 1.0]), (len(w), 1))
    phase = np.ones((nphamat, *mu.shape))

    return make_aer_core_v2(
        w=w * ureg.nm,
        phamat=phamat_names[:nphamat],
        mu=mu * ureg.dimensionless,
        theta=np.degrees(np.arccos(mu)) * ureg.deg,
        ext=np.ones_like(w) * ureg("km^-1"),
        ssa=np.ones_like(w) * ureg.dimensionless,
        phase=phase * ureg("1/sr"),
        normalize=True,
    )


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

    def test_factory(self, modes_all_double):
        """
        Instantiation from a dict succeeds and returns a ParticlePhaseFunction.
        We also test the particle_properties parameter conversion protocol.
        """

        from eradiate.scenes.phase import phase_function_factory

        pp = phase_function_factory.convert(
            {
                "type": "particlephase",
                "particle_properties": "govaerts_2021-continental",
            }
        )
        assert isinstance(pp, ParticlePhaseFunction)

    def test_eval_impl(self, modes_all_double, pphase):
        # Repeated evaluations with the same wavelength variable do not trigger
        # a recomputation
        w = 550.0 * ureg.nm
        a = pphase._eval_impl(w)
        b = pphase._eval_impl(w)
        assert a is b

        # Evaluation with a different variables does trigger a recomputation
        c = pphase._eval_impl(550.0 * ureg.nm)
        assert a is not c
        np.testing.assert_array_equal(a[0], c[0])
        np.testing.assert_array_equal(a[1], c[1])

    def test_eval_mu(self, modes_all_double, pphase):
        si = SpectralIndex.new()
        result = pphase.eval_mu(si)
        pp = pphase.particle_properties
        n_iangle = pp.data.sizes["iangle"]
        n_mu_expected = n_iangle if pp.has_fixed_mu_grid else n_iangle * 2
        assert result.size == n_mu_expected

        # Repeated evaluations with the same wavelength value do not trigger a
        # recomputation
        assert pphase.eval_mu(si) is result

    def test_eval_phase(self, modes_all_double, pphase):
        si = SpectralIndex.new()
        result = pphase.eval_phase(si)
        pp = pphase.particle_properties
        n_phamat = pp.data.sizes["phamat"]
        n_iangle = pp.data.sizes["iangle"]
        n_mu_expected = n_iangle if pp.has_fixed_mu_grid else n_iangle * 2
        assert result.shape == (n_phamat, n_mu_expected)

        # Repeated evaluations with the same wavelength value do not trigger a
        # recomputation
        assert pphase.eval_phase(si) is result

    @pytest.mark.parametrize(
        "nphamat, expected",
        [
            (4, {"m11": 0, "m12": 1, "m33": 2, "m34": 3, "m22": 0, "m44": 2}),
            (6, {"m11": 0, "m12": 1, "m33": 2, "m34": 3, "m22": 4, "m44": 5}),
        ],
        ids=["spherical", "spheroidal"],
    )
    def test_param_to_phamat(self, mode_ckd_polarized, nphamat, expected):
        """
        Phase matrix components map to the storage indices documented in
        eval_phase(), and those indices are within bounds.
        """
        pphase = ParticlePhaseFunction(
            particle_properties=ParticleProperties(make_minimal_aer_core_v2(nphamat))
        )
        assert pphase._param_to_phamat() == expected

        si = SpectralIndex.new()
        for index in expected.values():
            assert pphase.eval_phase(si, index).ndim == 1

    def test_eval_pmom(self, modes_all_double, pphase):
        si = SpectralIndex.new()
        if pphase.particle_properties.pmom is None:
            with pytest.raises(ValueError, match="No Legendre moments"):
                pphase.eval_pmom(si)
        else:
            result = pphase.eval_pmom(si)
            assert result.shape == (pphase.particle_properties.data.sizes["imom"],)
