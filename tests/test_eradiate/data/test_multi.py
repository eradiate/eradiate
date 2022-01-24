import pytest

import eradiate
from eradiate.data import DirectoryDataStore, MultiDataStore, OnlineDataStore
from eradiate.exceptions import DataError

TEST_STORE = "http://eradiate.eu/data/store"
TEST_FILE_SUBMODULE_REGISTERED = "tests/data/git/registered_dataset.nc"
TEST_FILE_SUBMODULE_UNREGISTERED = "tests/data/git/unregistered_dataset.nc"
TEST_FILE_ONLINE_REGISTERED = "tests/data/online/registered_dataset.nc"
TEST_FILE_ONLINE_UNREGISTERED = "tests/data/online/unregistered_dataset.nc"


def test_multi_data_store(tmpdir):
    # Initialise test data store
    data_store = MultiDataStore(
        [
            DirectoryDataStore(path=eradiate.config.dir / "resources/data"),
            OnlineDataStore(base_url=TEST_STORE, path=tmpdir),
        ]
    )

    # Fetch files from both data stores
    assert data_store.fetch(TEST_FILE_SUBMODULE_REGISTERED)
    assert data_store.fetch(TEST_FILE_ONLINE_REGISTERED)

    # Unregistered data will go through only if they are available from the
    # online store
    assert data_store.fetch(TEST_FILE_ONLINE_UNREGISTERED)
    with pytest.raises(DataError):
        data_store.fetch(TEST_FILE_SUBMODULE_UNREGISTERED)
