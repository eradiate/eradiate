from pathlib import Path

import pytest

import eradiate
from eradiate.data import BlindDirectoryDataStore
from eradiate.exceptions import DataError

TEST_FILE_EXISTS = Path("tests/data/git/registered_dataset.nc")
TEST_FILE_DOES_NOT_EXIST = Path("tests/data/git/does_not_exist.nc")


def test_directory_data_store_fetch():
    # The data submodule can be instantiated
    data_store = BlindDirectoryDataStore(
        path=eradiate.config.source_dir / "resources/data"
    )

    # We can fetch the test file
    assert data_store.fetch(TEST_FILE_EXISTS)

    # A missing file raises
    with pytest.raises(DataError):
        data_store.fetch(TEST_FILE_DOES_NOT_EXIST)
