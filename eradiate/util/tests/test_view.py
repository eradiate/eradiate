import numpy as np
import pytest
import xarray as xr

import enoki as ek
import eradiate.util.view as view


def test_bsdf_wrapper(variant_scalar_mono):
    """Test the Mitsuba BSDF-Plugin wrapper, by instantiating the wrapper class
    and calling its evaluate method with different values"""
    from eradiate.kernel.core.xml import load_dict

    bsdf_dict = {
        'type'       : 'diffuse',
        "reflectance": {"type": "uniform", "value": 0.5}
    }

    with pytest.raises(TypeError):
        wrapper = view.MitsubaBSDFPluginAdapter(bsdf_dict)

    bsdf = load_dict(bsdf_dict)
    bsdf_wrapper = view.MitsubaBSDFPluginAdapter(bsdf)

    assert ek.allclose(bsdf_wrapper.evaluate(
        [0, 0, 1], [1, 1, 1], 550), 0.15915, atol=1e-4)
    assert ek.allclose(bsdf_wrapper.evaluate(
        [30, 60], [0, 0, 1], 550), 0.15915, atol=1e-4)


def make_dataarray(sza, saa, vza, vaa, wavelength):
    data = np.random.rand(len(sza), len(saa), len(
        vza), len(vaa), len(wavelength))
    array = xr.DataArray(data, coords=[sza, saa, vza, vaa, wavelength], dims=[
        'sza', 'saa', 'vza', 'vaa', 'wavelength'])

    return array


def test_plane(variant_scalar_mono):
    """Test the plane method by creating a plane view from a DataArray and comparing it to a reference
    """

    arr = xr.DataArray([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]],
                       dims=["theta_o", "phi_o"],
                       coords={"theta_o": [0, 30, 60, 90], "phi_o": [0, 90, 180, 270]})

    arr.theta_o.attrs["units"] = "deg"
    arr.phi_o.attrs["units"] = "deg"
    arr.attrs["angular_type"] = "intrinsic"
    plane = view.plane(arr, phi=90)

    assert plane.attrs["angular_type"] == "intrinsic"
    assert np.all(plane["theta_o"] == [-90, -60, -30, 0, 30, 60, 90])
    assert np.all(plane.values == [16, 12, 8, 2, 6, 10, 14])


def test_pplane(variant_scalar_mono):
    """Test the pplane convenience function by comparing it to the result of manually calling
    plane with the corresponding arguments."""

    data = np.random.rand(4, 4, 4, 4)

    arr = xr.DataArray(data, dims=["theta_i", "phi_i", "theta_o", "phi_o"],
                       coords={"theta_i": [0, 30, 60, 90], "phi_i": [0, 90, 180, 270],
                               "theta_o": [0, 30, 60, 90], "phi_o": [0, 90, 180, 270]})
    arr.theta_o.attrs["units"] = "deg"
    arr.phi_o.attrs["units"] = "deg"
    arr.theta_i.attrs["units"] = "deg"
    arr.phi_i.attrs["units"] = "deg"
    arr.attrs["angular_type"] = "intrinsic"

    pplane = view.pplane(arr, theta_i=60, phi_i=90)
    hdata = arr.ert.sel(theta_i=60, phi_i=90)
    plane = view.plane(hdata, phi=90)

    assert np.all(pplane == plane)


def test_accessor_angular_dimensions(variant_scalar_mono):
    """Test the checks for angular dimensions, by creating DataArrays with different
    namings and asserting the correct number of angular dimensions is recognized.
    """

    dims = ["x", "y", "z", "w"]
    arr = xr.DataArray(np.zeros((3, 3, 3, 3)), dims=dims,
                       coords={"x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3], "w": [1, 2, 3]})

    # bi-hemispherical array
    arr.x.attrs["units"] = "deg"
    arr.y.attrs["units"] = "degrees"
    arr.z.attrs["units"] = "rad"
    arr.w.attrs["units"] = "radians"

    assert arr.ert._num_angular_dimensions() == 4
    assert arr.ert.is_bihemispherical()
    assert not arr.ert.is_hemispherical()

    # array without angular dimensions
    arr.x.attrs["units"] = "degert"
    arr.y.attrs["units"] = "angle"
    arr.z.attrs["units"] = "rod"
    arr.w.attrs["units"] = "radian"

    assert arr.ert._num_angular_dimensions() == 0
    assert not arr.ert.is_bihemispherical()
    assert not arr.ert.is_hemispherical()

    # hemispherical array
    arr.x.attrs["units"] = "deg"
    arr.y.attrs["units"] = "angle"
    arr.z.attrs["units"] = "rad"
    arr.w.attrs["units"] = "radian"

    assert arr.ert._num_angular_dimensions() == 2
    assert not arr.ert.is_bihemispherical()
    assert arr.ert.is_hemispherical()


def test_accessor_sel(variant_scalar_mono):
    """Test the sel method wrapper by creating DataArrays with different angular data types.
    Each array is accessed with dimension names from the other supported type and
    the correct reaction is asserted. An unsupported type is instantiated and the
    raising of correct exceptions is asserted.
    """

    data = np.ones((2, 2))
    arr_local = xr.DataArray(data, dims=["theta_i", "phi_i"],
                             coords={"theta_i": [0, 90], "phi_i": [0, 180]})
    arr_local.attrs["angular_type"] = "intrinsic"

    arr_eo = xr.DataArray(data, dims=["sza", "saa"],
                          coords={"sza": [0, 90], "saa": [0, 180]})
    arr_eo.attrs["angular_type"] = "observation"

    arr_wrong = xr.DataArray(data, dims=["hans", "helm"],
                             coords={"hans": [0, 90], "helm": [0, 180]})
    arr_wrong.attrs["angular_type"] = "weird"

    arr_missing = xr.DataArray(data, dims=["sza", "saa"],
                               coords={"sza": [0, 90], "saa": [0, 180]})

    assert arr_local.ert.sel(theta_i=0, phi_i=0).values == [[1]]

    with pytest.raises(ValueError) as exc:
        arr_local.ert.sel(sza=0, saa=0)

    assert arr_eo.ert.sel(sza=0, saa=0).values == arr_eo.ert.sel(theta_i=0, phi_i=0).values

    with pytest.raises(KeyError) as exc:
        arr_wrong.ert.sel(theta_o=0, phi_o=0)
    assert "Unknown angular type: weird" in str(exc.value)

    with pytest.raises(KeyError) as exc:
        arr_missing.ert.sel(theta_o=0, phi_o=0)
    assert "No angular data type was set. Cannot identify data nomenclature." in str(exc.value)
