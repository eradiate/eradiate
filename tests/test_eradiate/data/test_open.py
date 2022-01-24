from eradiate import data

TEST_FILE_SUBMODULE = "tests/data/git/registered_dataset.nc"
TEST_FILE_ONLINE = "tests/data/online/registered_dataset.nc"


def test_open_dataset():
    # Try opening a dataset from the submodule
    with data.open_dataset(TEST_FILE_SUBMODULE):
        assert True

    # Try opening a very small dataset from the online store
    with data.open_dataset(TEST_FILE_ONLINE):
        assert True


def test_load_dataset():
    assert data.load_dataset(TEST_FILE_SUBMODULE)
