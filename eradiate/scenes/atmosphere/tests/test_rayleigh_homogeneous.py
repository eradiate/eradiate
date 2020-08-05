import numpy as np
import pytest

import eradiate
from eradiate.scenes.core import KernelDict
from eradiate.scenes.atmosphere.rayleigh import (
    _LOSCHMIDT, _IOR_DRY_AIR, kf, sigma_s_single,
    sigma_s_mixture, delta, RayleighHomogeneousAtmosphere
)
from eradiate.util.collections import onedict_value
from eradiate.util.units import Q_


def test_king_correction_factor():
    """Test computation of King correction factor"""

    # Compare default mean depolarisation ratio for dry air given by
    # (Young, 1980) with corresponding value
    assert np.allclose(kf(0.0279), 1.048, rtol=1.e-2)


def test_sigma_s_single():
    """Test computation of Rayleigh scattering coefficient with default values"""

    ref_cross_section = Q_(4.513e-27, "cm**2")
    ref_sigmas = ref_cross_section * _LOSCHMIDT
    expected = ref_sigmas.to("km^-1").magnitude

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    assert np.allclose(sigma_s_single(), expected, rtol=1e-2)


def test_sigma_s_mixture():
    """Test computation of the Rayleigh scattering coefficient for a mixture of
    particles types by calling the function with the parameters for a single
    particle type, namely air particles, then for a mixture of two particle
    types.
    """
    coefficient_air = \
        sigma_s_mixture(
            550.,
            [_LOSCHMIDT.magnitude],
            _IOR_DRY_AIR.magnitude,
            [_LOSCHMIDT.magnitude],
            [_IOR_DRY_AIR.magnitude],
            [1.049]
        )

    assert np.allclose(coefficient_air, sigma_s_single(), rtol=1e-6)

    coefficient_2_particle_types_mixture = \
        sigma_s_mixture(
            550.,
            _LOSCHMIDT.magnitude * np.ones(2) / 2,
            2.,
            _LOSCHMIDT.magnitude * np.ones(2),
            2. * np.ones(2),
            1. * np.ones(2)
        )
    expected_value = 24 * np.pi ** 3 / ((550.e-12) ** 4 * _LOSCHMIDT.magnitude)

    assert np.allclose(
        coefficient_2_particle_types_mixture,
        expected_value,
        rtol=1e-6
    )


def test_delta():
    assert np.isclose(delta(), 0.9587257754327136, rtol=1e-6)


@pytest.mark.parametrize("ref", (False, True))
def test_rayleigh_homogeneous(mode_mono, ref):
    # This test checks the functionality of RayleighHomogeneousAtmosphere

    from eradiate.kernel.core.xml import load_dict

    # Check if default constructor works
    r = RayleighHomogeneousAtmosphere()

    # Check if default constructs can be loaded by the kernel
    dict_phase = onedict_value(r.phase())
    assert load_dict(dict_phase) is not None

    dict_medium = onedict_value(r.media())
    assert load_dict(dict_medium) is not None

    dict_shape = onedict_value(r.shapes())
    assert load_dict(dict_shape) is not None

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(r)
    assert kernel_dict.load() is not None

    # Construct with parameters
    eradiate.mode.config["wavelength"] = 650.
    r = RayleighHomogeneousAtmosphere({"height": 10.})

    # check if sigma_s was correctly computed using the mode wavelength value
    assert np.isclose(r._sigma_s,
                      sigma_s_single(wavelength=eradiate.mode.config["wavelength"]))

    # check if automatic scene width works as intended
    assert np.isclose(r._width,
                      10. / sigma_s_single(wavelength=eradiate.mode.config["wavelength"]))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(r).load() is not None

    # Check that sigma_s wavelength specification is overridden
    eradiate.mode.config["wavelength"] = 650.
    r = RayleighHomogeneousAtmosphere({"sigma_s": 600.})
    assert r._sigma_s != sigma_s_single(wavelength=600.)

    # Check that we can set sigma_s_params with the other parameters of
    # sigma_s_single
    params = {
        "number_density": 2.0e34,
        "refractive_index": 1.0003,
        "king_factor": 1.05
    }
    r = RayleighHomogeneousAtmosphere({"sigma_s": params})
    assert r._sigma_s == sigma_s_single(
        wavelength=eradiate.mode.config["wavelength"], **params
    )
