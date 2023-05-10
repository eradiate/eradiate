import os.path
from datetime import datetime
from pathlib import Path

import pytest
import xarray as xr

from eradiate.data import SafeOnlineDataStore
from eradiate.exceptions import DataError

TEST_STORE = "http://eradiate.eu/data/store/stable"
TEST_FILE = Path("tests/data/online/registered_dataset.nc")


def test_online_data_store_registry(tmpdir):
    # Using an empty storage directory, a SafeOnlineDataStore instance can be created
    store = SafeOnlineDataStore(base_url=TEST_STORE, path=tmpdir)

    # The registry file is downloaded
    registry_filename = store.registry_fetch()
    assert registry_filename.is_file()

    # Reloading the registry works as expected
    store.registry["foo"] = "bar"
    store.registry_reload()
    assert "foo" not in store.registry

    # We can force a registry download
    last_state = store.registry.copy()
    last_modified = datetime.fromtimestamp(registry_filename.stat().st_mtime)
    store.registry_reload(delete=True)
    # The registry contents are the same but the file is more recent
    assert store.registry == last_state
    assert store.registry is not last_state
    assert datetime.fromtimestamp(registry_filename.stat().st_mtime) > last_modified


def test_oneline_data_store_is_registered(tmpdir):
    store = SafeOnlineDataStore(base_url=TEST_STORE, path=tmpdir)

    # Default behaviour allows matching compressed files
    filename = store.is_registered(TEST_FILE)
    assert str(filename) == str(TEST_FILE) + ".gz"

    # The test file is actually gzip-compressed and will therefore not match a
    # registry entry if allow_compressed is set to False
    with pytest.raises(ValueError):
        store.is_registered(TEST_FILE, allow_compressed=False)


def test_online_data_store_fetch(tmpdir):
    store = SafeOnlineDataStore(base_url=TEST_STORE, path=tmpdir)
    filename = os.path.join(tmpdir, TEST_FILE)

    # Upon calling fetch(), gzip detection is successful: the gz file is downloaded
    assert not os.path.isfile(filename + ".gz")
    store.fetch(TEST_FILE)
    assert os.path.isfile(filename + ".gz")

    # The gz file is also decompressed
    assert os.path.isfile(filename)
    with xr.open_dataset(store.fetch(TEST_FILE)) as ds:
        assert isinstance(ds, xr.Dataset)

    # A direct and blind download is attempted if the file is not registered
    # In this specific case, it will fail because the file does not exist on the
    # server
    with pytest.raises(DataError):
        store.fetch("foo")


def test_online_data_store_purge(tmpdir):
    # We start with an empty directory
    assert not any(os.scandir(tmpdir))

    # We create a data store and fetch a file:
    # we expect the directory to contain exactly 2 elements (the registry file
    # and the directory containing the fetched file)
    store = SafeOnlineDataStore(base_url=TEST_STORE, path=tmpdir)
    store.fetch(TEST_FILE)
    assert len(list(os.scandir(tmpdir))) == 2
    assert store.registry_path.is_file()
    assert (store.path / TEST_FILE).is_file()

    # We purge the directory but keep registered files
    os.makedirs(store.path / "foo")  # This empty directory should also be cleaned up
    store.purge(keep="registered")
    assert len(list(os.scandir(tmpdir))) == 2  # We still have registry.txt and a subdir
    assert store.registry_path.is_file()
    assert not (store.path / TEST_FILE).is_file()  # The decompressed file is gone
    assert (
        store.path / (str(TEST_FILE) + ".gz")
    ).is_file()  # The compressed file is still here

    # Purge everything: all files are removed
    store.purge()
    assert len(list(os.scandir(tmpdir))) == 0
