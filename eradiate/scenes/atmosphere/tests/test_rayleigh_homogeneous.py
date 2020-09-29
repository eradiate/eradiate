import numpy as np
import pytest

import eradiate
from eradiate.scenes.atmosphere.homogeneous import (
    _LOSCHMIDT, RayleighHomogeneousAtmosphere,
    kf, sigma_s_single
)
from eradiate.scenes.core import KernelDict
from eradiate.util.collections import onedict_value
from eradiate.util.units import config_default_units, ureg


def test_king_correction_factor():
    """Test computation of King correction factor"""

    # Compare default mean depolarisation ratio for dry air given by
    # (Young, 1980) with corresponding value
    assert np.allclose(kf(0.0279), 1.048, rtol=1.e-2)


def test_sigma_s_single():
    """Test computation of Rayleigh scattering coefficient with default values"""

    ref_cross_section = ureg.Quantity(4.513e-27, "cm**2")
    ref_sigmas = ref_cross_section * _LOSCHMIDT
    expected = ref_sigmas

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    print(expected.to("m^-1"))
    print(sigma_s_single().to("m^-1"))
    assert np.allclose(sigma_s_single(), expected, rtol=1e-2)


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
    eradiate.mode.config["wavelength_unit"] = config_default_units.get("wavelength")
    r = RayleighHomogeneousAtmosphere({"height": 10.})

    # check if sigma_s was correctly computed using the mode wavelength value
    wavelength = ureg.Quantity(eradiate.mode.config["wavelength"], eradiate.mode.config["wavelength_unit"])
    assert np.isclose(r._sigma_s, sigma_s_single(wavelength=wavelength))

    # check if automatic scene width works as intended
    assert np.isclose(r._width, 10. / sigma_s_single(wavelength=wavelength))

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
        "number_density_unit": "km^-3",
        "refractive_index": 1.0003,
        "king_factor": 1.05
    }
    params_pint = {
        "number_density": params["number_density"] * ureg(params["number_density_unit"]),
        "refractive_index": params["refractive_index"],
        "king_factor": params["king_factor"]
    }
    r = RayleighHomogeneousAtmosphere({"sigma_s": params})
    assert r._sigma_s == sigma_s_single(
        wavelength=eradiate.mode.config["wavelength"] * eradiate.mode.config["wavelength_unit"], **params_pint
    )
