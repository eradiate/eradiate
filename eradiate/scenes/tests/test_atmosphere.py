import numpy as np
import pytest

from eradiate.scenes import SceneDict
from eradiate.scenes.atmosphere.rayleigh import (
    _LOSCHMIDT, _IOR_DRY_AIR, king_factor, sigmas_single,
    sigmas_mixture, delta, RayleighHomogeneous
)
from eradiate.util.units import Q_


def test_king_correction_factor():
    """Test computation of King correction factor"""

    # Compare to a reference value
    assert np.allclose(king_factor(), 1.048, rtol=1.e-2)


def test_sigmas_single():
    """Test computation of Rayleigh scattering coefficient with default values"""

    ref_cross_section = Q_(4.513e-27, 'cm**2')
    ref_sigmas = ref_cross_section * _LOSCHMIDT
    expected = ref_sigmas.to('m^-1').magnitude

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    assert np.allclose(sigmas_single(), expected, rtol=1e-2)


def test_sigmas_mixture():
    """Test computation of the Rayleigh scattering coefficient for a mixture of
    particles types by calling the function with the parameters for a single
    particle type, namely air particles, then for a mixture of two particle
    types.
    """
    coefficient_air = \
        sigmas_mixture(
            550., 
            [_LOSCHMIDT.magnitude], 
            _IOR_DRY_AIR.magnitude,
            [_LOSCHMIDT.magnitude],
            [_IOR_DRY_AIR.magnitude],
            [1.049]
        )

    assert np.allclose(coefficient_air, sigmas_single(), rtol=1e-6)

    coefficient_2_particle_types_mixture = \
        sigmas_mixture(
            550.,
            _LOSCHMIDT.magnitude * np.ones(2) / 2,
            2.,
            _LOSCHMIDT.magnitude * np.ones(2),
            2. * np.ones(2),
            1. * np.ones(2)
        )
    expected_value = 24 * np.pi ** 3 / ((550.e-9) ** 4 * _LOSCHMIDT.magnitude)

    assert np.allclose(
        coefficient_2_particle_types_mixture,
        expected_value,
        rtol=1e-6
    )


def test_delta():
    assert np.isclose(delta(), 0.9587257754327136, rtol=1e-6)


@pytest.mark.parametrize("ref", (False, True))
def test_rayleigh_homogeneous(variant_scalar_mono, ref):
    from eradiate.kernel.core.xml import load_dict
    # Default constructor
    r = RayleighHomogeneous()

    dict_phase = next(iter(r.phase().values()))
    assert load_dict(dict_phase) is not None

    dict_medium = next(iter(r.media().values()))
    assert load_dict(dict_medium) is not None

    dict_shape = next(iter(r.shapes().values()))
    assert load_dict(dict_shape) is not None

    # Check if produced scene can be instantiated
    scene_dict = SceneDict.empty()
    scene_dict.add(r)
    assert scene_dict.load() is not None

    # Construct with parameters
    r = RayleighHomogeneous(dict(
        height=10.,
        sigmas_params={"wavelength": 550.}
    ))

    # Check if produced scene can be instantiated
    assert SceneDict.empty().add(r).load() is not None
