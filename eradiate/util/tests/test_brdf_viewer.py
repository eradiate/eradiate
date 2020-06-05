import enoki as ek
import numpy as np
import pytest
import xarray as xr

import eradiate.util.brdf_viewer as bv


def test_bsdf_wrapper(variant_scalar_mono):
    """Test the Mitsuba BSDF-Plugin wrapper, by instantiating the wrapper class
    and calling its evaluate method with different values"""
    from eradiate.kernel.core.xml import load_dict

    bsdf_dict = {
        'type': 'diffuse',
        "reflectance": {"type": "uniform", "value": 0.5}
    }

    with pytest.raises(TypeError):
        wrapper = bv.MitsubaBSDFPluginAdapter(bsdf_dict)

    bsdf = load_dict(bsdf_dict)
    bsdf_wrapper = bv.MitsubaBSDFPluginAdapter(bsdf)

    assert ek.allclose(bsdf_wrapper.evaluate(
        [0, 0, 1], [1, 1, 1], 550), 0.15915, atol=1e-4)
    assert ek.allclose(bsdf_wrapper.evaluate(
        [30, 60], [0, 0, 1], 550), 0.15915, atol=1e-4)


def make_xarray(theta_i, phi_i, theta_o, phi_o, wavelength):
    data = np.random.rand(len(theta_i), len(phi_i), len(
        theta_o), len(phi_o), len(wavelength))
    array = xr.DataArray(data, coords=[theta_i, phi_i, theta_o, phi_o, wavelength], dims=[
                         'theta_i', 'phi_i', 'theta_o', 'phi_o', 'wavelength'])

    return array


def test_xarray_wrapper(variant_scalar_mono):
    """Testing the XArray wrapper, by instantiating the wrapper class and calling its
    evaluate method with different values"""

    array = make_xarray([0, 45, 90], [0, 180, 360], [
                        0, 45, 90], [0, 180, 360], [550])

    with pytest.raises(TypeError):
        wrapper = bv.DataArrayBRDFAdapter("test")

    wrapper = bv.DataArrayBRDFAdapter(array)

    num_tests = 0
    for i in range(num_tests):
        rands = np.random.rand(2)
        theta_i = 45
        phi_i = 0
        theta_o = rands[0] * 90
        phi_o = rands[1] * 360

        assert np.all(wrapper.evaluate([theta_o, phi_o], [theta_i, phi_i], [550]) == array.sel(theta_i=theta_i, phi_i=phi_i, wavelength=550).interp(
            theta_o=theta_o, phi_o=phi_o, kwargs={'fill_value': 0.0}).data)

    # wavelength cannot be interpolated, so it must be called with the 
    # exact value of a data point
    with pytest.raises(KeyError) as e:
        wrapper.evaluate([45, 180], [0, 45], [600])
