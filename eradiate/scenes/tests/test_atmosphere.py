import numpy as np

from eradiate.scenes.atmosphere import (
    _LOSCHMIDT, RayleighHomogeneous, king_correction_factor, rayleigh_delta,
    rayleigh_scattering_coefficient_1, rayleigh_scattering_coefficient_mixture
)
from eradiate.util import Q_


def test_king_correction_factor():
    """Test computation of King correction factor"""

    # Compare to a reference value
    assert np.allclose(king_correction_factor(), 1.048, rtol=1.e-2)


def test_rayleigh_scattering_coefficient_1():
    """Test computation of Rayleigh scattering coefficient with default values"""

    reference_scattering_cross_section = Q_(4.513e-27, 'cm**2')
    reference_scattering_coefficient = reference_scattering_cross_section * _LOSCHMIDT
    reference_value = reference_scattering_coefficient.to('m^-1').magnitude

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    assert np.allclose(rayleigh_scattering_coefficient_1(),
                       reference_value, rtol=1e-2)


def test_rayleigh_scattering_coefficient_mixture():
    """Test computation of the Rayleigh scattering coefficient for a mixture of
    particles types by calling the function with the parameters for a single
    particle type, namely air particles, then for a mixture of two particle
    types.
    """
    coefficient_air = \
        rayleigh_scattering_coefficient_mixture(
            550, [_LOSCHMIDT.magnitude], 1.0002932,
            [_LOSCHMIDT.magnitude],
            [1.0002932],
            [1.049]
        )

    assert np.allclose(
        coefficient_air, rayleigh_scattering_coefficient_1(), rtol=1e-6)

    coefficient_2_particle_types_mixture = \
        rayleigh_scattering_coefficient_mixture(
            550.,
            _LOSCHMIDT.magnitude * np.ones(2) / 2,
            2.,
            _LOSCHMIDT.magnitude * np.ones(2),
            2. * np.ones(2),
            1. * np.ones(2)
        )
    expected_value = 24 * np.pi**3 / ((550.e-9)**4 * _LOSCHMIDT.magnitude)

    assert np.allclose(
        coefficient_2_particle_types_mixture,
        expected_value,
        rtol=1e-6
    )


def test_rayleigh_delta():
    assert np.isclose(rayleigh_delta(),
                      0.9587257754327136, rtol=1e-6)


def test_rayleigh_homogeneous():
    # Default constructor
    r = RayleighHomogeneous()
    
    assert r.phase()[0].to_xml() == \
        '<phase type="rayleigh" id="phase_rayleigh"/>'
    assert r.media()[0].to_xml() == \
        f'<medium type="homogeneous" id="medium_rayleigh">' \
        f'<ref id="phase_rayleigh"/>' \
        f'<spectrum name="sigma_t" value="{rayleigh_scattering_coefficient_1()}"/>' \
        f'<spectrum name="albedo" value="1.0"/>' \
        f'</medium>'
    assert r.shapes()[0].to_xml() == \
        '<shape type="cube">' \
        '<transform name="to_world">' \
        '<scale value="1.0, 1.0, 1.0"/>' \
        '<translate value="0.0, 0.0, 1.0"/>' \
        '</transform>' \
        '<ref name="interior" id="medium_rayleigh"/>' \
        '<bsdf type="null"/>' \
        '</shape>'
