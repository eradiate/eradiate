import numpy as np
import pytest
import xarray as xr

from eradiate.util.xarray import (
    CoordSpecRegistry, DatasetSpec, VarSpec, make_dataarray, plane,
    pplane, validate_metadata, CoordSpec
)


def test_plane():
    """Test the plane method by creating a plane view from a DataArray and
    comparing it to a reference."""

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])

    arr = xr.DataArray(
        data,
        dims=["theta_o", "phi_o"],
        coords={
            "theta_o": [0, 30, 60, 90],
            "phi_o": [0, 90, 180, 270]
        }
    )

    p = plane(arr, phi=90)
    assert np.all(p["theta_o"] == [-90, -60, -30, 0, 30, 60, 90])
    assert np.all(p.values.squeeze() == [16, 12, 8, 2, 6, 10, 14])

    arr = xr.DataArray(
        data,
        dims=["theta", "phi"],
        coords={
            "theta": [0, 30, 60, 90],
            "phi": [0, 90, 180, 270]
        }
    )

    p = plane(arr, phi=90, theta_dim="theta", phi_dim="phi")
    assert np.all(p["theta"] == [-90, -60, -30, 0, 30, 60, 90])
    assert np.all(p.values.squeeze() == [16, 12, 8, 2, 6, 10, 14])


def test_pplane():
    """Test the pplane convenience function by comparing it to the result of
    manually calling plane with the corresponding arguments."""

    data = np.random.rand(4, 4, 4, 4)

    arr = xr.DataArray(
        data,
        dims=["vza", "vaa", "sza", "saa"],
        coords={
            "vza": [0, 30, 60, 90],
            "vaa": [0, 90, 180, 270],
            "sza": [0, 30, 60, 90],
            "saa": [0, 90, 180, 270]
        }
    )
    pp = pplane(arr, sza=60, saa=90)
    p = plane(arr.sel(sza=60, saa=90), phi=90, theta_dim="vza", phi_dim="vaa")

    assert np.all(pp == p)


def test_make_dataarray():
    # Check that basic array creation works
    da = make_dataarray(
        data=np.ones((1, 1, 1, 1, 1)),
        coords={
            "vza": [0.],
            "vaa": [0.],
            "sza": [0.],
            "saa": [0.],
            "wavelength": [500.]
        },
        dims=("vza", "vaa", "sza", "saa", "wavelength"),
        var_spec=VarSpec(coord_specs="angular_observation"),
    )

    assert da.dims == ("vza", "vaa", "sza", "saa", "wavelength")
    assert set(da.coords.keys()) == {
        "vza", "vaa", "sza", "saa", "wavelength"
    }

    # Check that metadata are applied properly if required
    da = make_dataarray(
        data=np.ones((1, 1, 1, 1, 1)),
        coords={
            "vza": [0.],
            "vaa": [0.],
            "sza": [0.],
            "saa": [0.],
            "wavelength": [500.]
        },
        dims=("vza", "vaa", "sza", "saa", "wavelength"),
        var_spec=VarSpec(coord_specs="angular_observation"),
    )

    assert da.coords["vza"].attrs == {
        "standard_name": "sensor_zenith_angle",
        "units": "deg",
        "long_name": "sensor zenith angle"
    }


def test_spec():
    # VarSpec: check if parameter consistency is properly enforced
    with pytest.raises(ValueError):
        VarSpec(standard_name="foo")
    with pytest.raises(ValueError):
        VarSpec(standard_name="foo", units="m")

    # DatasetSpec: check if parameter consistency is properly enforced
    with pytest.raises(ValueError):
        DatasetSpec(title="", history="", references="")


@pytest.fixture
def dataarray_without_metadata():
    return xr.DataArray(
        data=np.random.random((1, 1, 1, 1, 1)),
        dims=["vza", "vaa", "sza", "saa", "wavelength"],
        coords={
            "vza": [0.],
            "vaa": [0.],
            "sza": [0.],
            "saa": [0.],
            "wavelength": [500.]
        }
    )


@pytest.fixture
def dataset_without_metadata():
    return xr.Dataset(
        data_vars={
            "p": (("z_layer", "z_level", "species"), np.random.random((1, 1, 1))),
        },
        coords={
            "z_layer": ("z_layer", [0.]),
            "z_level": ("z_level", [0.]),
            "species": ("species", ["foo"])
        },
    )


def test_validate_metadata(dataarray_without_metadata):
    dataarray = dataarray_without_metadata

    # Check and apply missing metadata to a single coordinate
    coord_spec = CoordSpecRegistry.get("vza")
    attrs = validate_metadata(dataarray.vza, coord_spec, normalize=True)
    assert attrs == {
        "standard_name": "sensor_zenith_angle",
        "units": "deg",
        "long_name": "sensor zenith angle"
    }


def test_dataarray_accessor_validate_metadata(dataarray_without_metadata):
    dataarray = dataarray_without_metadata
    var_spec = VarSpec(
        standard_name="toa_brf",
        units="",
        long_name="top-of-atmosphere BRF",
        coord_specs="angular_observation"
    )

    # Check that the validator complains
    with pytest.raises(ValueError):
        dataarray.ert.validate_metadata(var_spec)

    # Check and apply missing metadata
    dataarray.ert.normalize_metadata(var_spec)
    assert dataarray.attrs == {
        "standard_name": "toa_brf",
        "units": "",
        "long_name": "top-of-atmosphere BRF"
    }
    assert dataarray.sza.attrs == {
        "standard_name": "solar_zenith_angle",
        "units": "deg",
        "long_name": "solar zenith angle"
    }


def test_dataset_accessor_validate_metadata(dataset_without_metadata):
    dataset = dataset_without_metadata
    dataset_spec = DatasetSpec(
        convention="CF-1.8",
        title="My awesome test dataset",
        history="None",
        references="None",
        source="Eradiate test suite",
        var_specs={
            "p": VarSpec(standard_name="pressure", units="Pa", long_name="air pressure")
        },
        coord_specs="atmospheric_profile"
    )

    # Check that the validator complains
    with pytest.raises(ValueError):
        dataset.ert.validate_metadata(dataset_spec)

    # Check and apply missing metadata
    dataset.ert.normalize_metadata(dataset_spec)
    assert dataset.attrs == {
        "convention": "CF-1.8",
        "history": "None",
        "references": "None",
        "source": "Eradiate test suite",
        "title": "My awesome test dataset"
    }
    assert dataset.p.attrs == {
        "standard_name": "pressure",
        "units": "Pa",
        "long_name": "air pressure"
    }
    assert dataset.species.attrs == {
        "standard_name": "species",
        "long_name": "species"
    }

    # check dataset specifications can be created to allow unknown attributes,
    # variables and coordinates
    ds_spec = DatasetSpec(
        convention="CF-1.8",
        title="My awesome test dataset",
        history="None",
        references="None",
        source="Eradiate test suite",
        var_specs={
            "y": VarSpec(standard_name="my_variable",
                         units=None,
                         long_name="my variable")
        },
        coord_specs={
            "x": CoordSpec(standard_name="my_coordinate",
                           units=None,
                           long_name="my coordinate")
        }
    )
    ds_with_unknown_attr = xr.Dataset(
        data_vars={
            "y": ("x", [1, 2, 3], {
                "standard_name": "my_variable",
                "long_name": "my variable"})
        },
        coords={
            "x": ("x", [1, 2, 3], {
                "standard_name": "my_coordinate",
                "long_name": "my coordinate"})
        },
        attrs={
            "convention": "CF-1.8",
            "title": "My awesome test dataset",
            "history": "None",
            "references": "None",
            "source": "Eradiate test suite",
            "unknown": "attributes are allowed"
        }
    )
    ds_with_unknown_attr.ert.validate_metadata(ds_spec, allow_unknown=True)

    with pytest.raises(ValueError):
        ds_with_unknown_attr.ert.validate_metadata(ds_spec, allow_unknown=False)
