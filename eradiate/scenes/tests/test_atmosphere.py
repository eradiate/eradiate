import numpy as np
import pytest

from eradiate.scenes.core import KernelDict
from eradiate.scenes.atmosphere.rayleigh import (
    _LOSCHMIDT, _IOR_DRY_AIR, kf, sigma_s_single,
    sigma_s_mixture, delta, RayleighHomogeneousAtmosphere
)
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
def test_rayleigh_homogeneous(variant_scalar_mono, ref):
    # This test checks the functionality of RayleighHomogeneousAtmosphere

    from eradiate.kernel.core.xml import load_dict

    # Check if default constructor works
    r = RayleighHomogeneousAtmosphere()

    # Check if default constructs can be loaded by the kernel
    dict_phase = next(iter(r.phase().values()))
    assert load_dict(dict_phase) is not None

    dict_medium = next(iter(r.media().values()))
    assert load_dict(dict_medium) is not None

    dict_shape = next(iter(r.shapes().values()))
    assert load_dict(dict_shape) is not None

    # Check if produced scene can be instantiated
    kernel_dict = KernelDict.empty()
    kernel_dict.add(r)
    assert kernel_dict.load() is not None

    # Construct with parameters
    r = RayleighHomogeneousAtmosphere(dict(
        height=10.,
        sigma_s={"wavelength": 650.}
    ))

    # check if sigma_s was correctly computed using wavelength value in
    # sigma_s_params
    assert np.isclose(r._sigma_s, sigma_s_single(wavelength=650.))

    # check if automatic scene width works as intended
    assert np.isclose(r._width, 10. / sigma_s_single(wavelength=650.))

    # Check if produced scene can be instantiated
    assert KernelDict.empty().add(r).load() is not None

    # Check that if no value for sigma_s is provided, sigma_s_single is called
    # to set the value of _sigma_s
    r = RayleighHomogeneousAtmosphere()
    assert r._sigma_s == sigma_s_single()

    # Check that we can set sigma_s_params with every parameter of
    # sigma_s_single
    params = {
        "wavelength": 480.,
        "number_density": 2.0e25,
        "refractive_index": 1.0003,
        "king_factor": 1.05
    }
    r = RayleighHomogeneousAtmosphere({"sigma_s": params})
    assert r._sigma_s == sigma_s_single(**params)
