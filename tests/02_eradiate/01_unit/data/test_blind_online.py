import os
import time
from pathlib import Path

import pytest

from eradiate.data import BlindOnlineDataStore
from eradiate.exceptions import DataError

TEST_STORE = "http://eradiate.eu/data/store/stable"


def test_blind_data_store_fetch(tmpdir):
    test_file = Path("tests/data/online/unregistered_dataset.nc")

    # Using an empty storage directory, a BlindOnlineDataStore instance can be created
    store = BlindOnlineDataStore(base_url=TEST_STORE, path=tmpdir)

    # We can download a resource if it is available online
    fpath = store.fetch(test_file)
    assert fpath.is_file()

    # Downloaded files are reused if they exist
    last_modified = os.path.getmtime(fpath)
    os.remove(fpath)
    time.sleep(0.1)
    assert last_modified != os.path.getmtime(store.fetch(test_file))
    last_modified = os.path.getmtime(fpath)
    assert last_modified == os.path.getmtime(store.fetch(test_file))

    # Requesting missing resources raises
    with pytest.raises(DataError):
        store.fetch("foo")


@pytest.mark.parametrize("keep", [False, True])
def test_blind_data_store_purge(tmpdir, keep):
    store = BlindOnlineDataStore(base_url=TEST_STORE, path=tmpdir)

    files = [
        "tests/data/online/unregistered_dataset.nc.gz",
        "tests/data/online/registered_dataset.nc.gz",
    ]

    # Download files
    for file in files:
        store.fetch(file)
    assert len(list(os.scandir(tmpdir / "tests/data/online"))) == 2
    print(list(os.scandir(tmpdir / "tests/data/online")))

    if not keep:
        store.purge(keep=None)
        # All files are removed
        assert len(list(os.scandir(tmpdir))) == 0

    else:
        store.purge(keep=["tests/data/online/unregistered_dataset.nc.gz"])
        # Specified files are kept, the others are removed
        assert len(list(os.scandir(tmpdir))) == 1
        assert (tmpdir / "tests/data/online/unregistered_dataset.nc.gz").isfile()
        assert not (tmpdir / "tests/data/online/registered_dataset.nc.gz").isfile()
