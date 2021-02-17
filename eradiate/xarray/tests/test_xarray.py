import pytest

from eradiate.xarray.metadata import (
    CoordSpecRegistry,
    DatasetSpec,
    VarSpec,
    validate_metadata
)


def test_spec():
    # VarSpec: check if parameter consistency is properly enforced
    with pytest.raises(ValueError):
        VarSpec(standard_name="foo")
    with pytest.raises(ValueError):
        VarSpec(standard_name="foo", units="m")

    # DatasetSpec: check if parameter consistency is properly enforced
    with pytest.raises(ValueError):
        DatasetSpec(title="", history="", references="")


def test_validate_metadata(dataarray_without_metadata):
    dataarray = dataarray_without_metadata

    # Check and apply missing metadata to a single coordinate
    coord_spec = CoordSpecRegistry.get("vza")
    attrs = validate_metadata(dataarray.vza, coord_spec, normalize=True)
    assert attrs == {
        "standard_name": "viewing_zenith_angle",
        "units": "deg",
        "long_name": "viewing zenith angle"
    }
