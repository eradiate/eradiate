from pathlib import Path

import pytest

import eradiate
from eradiate.data import SafeDirectoryDataStore
from eradiate.exceptions import DataError

TEST_FILE_REGISTERED = Path("tests/data/git/registered_dataset.nc")
TEST_FILE_UNREGISTERED = Path("tests/data/git/unregistered_dataset.nc")


def test_directory_data_store_fetch():
    # The data submodule can be instantiated
    data_store = SafeDirectoryDataStore(
        path=eradiate.config.source_dir / "resources/data"
    )

    # We can fetch the test file
    assert data_store.fetch(TEST_FILE_REGISTERED)

    # Unregistered files cannot be fetched
    with pytest.raises(DataError):
        data_store.fetch(TEST_FILE_UNREGISTERED)
