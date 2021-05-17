import pytest
import xarray as xr

from eradiate.xarray.metadata import CoordSpec, DatasetSpec, VarSpec


def test_dataarray_accessor_validate_metadata(dataarray_without_metadata):
    dataarray = dataarray_without_metadata
    var_spec = VarSpec(
        standard_name="toa_brf",
        units="",
        long_name="top-of-atmosphere BRF",
        coord_specs="angular_observation",
    )

    # Check that the validator complains
    with pytest.raises(ValueError):
        dataarray.ert.validate_metadata(var_spec)

    # Check and apply missing metadata
    dataarray.ert.normalize_metadata(var_spec)
    assert dataarray.attrs == {
        "standard_name": "toa_brf",
        "units": "",
        "long_name": "top-of-atmosphere BRF",
    }
    assert dataarray.sza.attrs == {
        "standard_name": "solar_zenith_angle",
        "units": "deg",
        "long_name": "solar zenith angle",
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
        coord_specs="atmospheric_profile",
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
        "title": "My awesome test dataset",
    }
    assert dataset.p.attrs == {
        "standard_name": "pressure",
        "units": "Pa",
        "long_name": "air pressure",
    }
    assert dataset.species.attrs == {"standard_name": "species", "long_name": "species"}

    # check dataset specifications can be created to allow unknown attributes,
    # variables and coordinates
    ds_spec = DatasetSpec(
        convention="CF-1.8",
        title="My awesome test dataset",
        history="None",
        references="None",
        source="Eradiate test suite",
        var_specs={
            "y": VarSpec(
                standard_name="my_variable", units=None, long_name="my variable"
            )
        },
        coord_specs={
            "x": CoordSpec(
                standard_name="my_coordinate", units=None, long_name="my coordinate"
            )
        },
    )
    ds_with_unknown_attr = xr.Dataset(
        data_vars={
            "y": (
                "x",
                [1, 2, 3],
                {"standard_name": "my_variable", "long_name": "my variable"},
            )
        },
        coords={
            "x": (
                "x",
                [1, 2, 3],
                {"standard_name": "my_coordinate", "long_name": "my coordinate"},
            )
        },
        attrs={
            "convention": "CF-1.8",
            "title": "My awesome test dataset",
            "history": "None",
            "references": "None",
            "source": "Eradiate test suite",
            "unknown": "attributes are allowed",
        },
    )
    ds_with_unknown_attr.ert.validate_metadata(ds_spec, allow_unknown=True)

    with pytest.raises(ValueError):
        ds_with_unknown_attr.ert.validate_metadata(ds_spec, allow_unknown=False)
