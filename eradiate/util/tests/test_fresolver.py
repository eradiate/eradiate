import os
from pathlib import Path

import pytest

from eradiate.util.fresolver import FileResolver


@pytest.fixture()
def fresolver():
    # This small fixture will avoid side effects
    FileResolver().reset()
    yield FileResolver()
    # Teardown: Reset file resolver again
    FileResolver().reset()


def test_contains(fresolver):
    # Assuming default path, we check that the resolver is correctly initialised
    assert fresolver.contains(os.getcwd())
    assert fresolver.contains(".")

    # We check that nonexisting elements are not found
    assert not fresolver.contains("/")


def test_remove(fresolver):
    # Remove from item
    fresolver.remove(os.getcwd())
    assert len(fresolver) == 1

    # Remove from index
    fresolver.remove(0)
    assert len(fresolver) == 0

    # We expect a removal attempt with a nonexisting item to raise
    with pytest.raises(ValueError):
        fresolver.remove(os.getcwd())


def test_prepend_append(fresolver):
    # We start from an empty path list
    fresolver.clear()

    # Test prepend
    fresolver.prepend(".")
    assert fresolver.contains(Path.cwd())

    # We can't prepend a nonexisting directory
    with pytest.raises(ValueError):
        fresolver.prepend("some/random/path/which/doesnt/exist")

    # Test append
    fresolver.append("/")
    assert fresolver[1] == Path("/")

    # We can't append a nonexisting directory
    with pytest.raises(ValueError):
        fresolver.append("some/random/path/which/doesnt/exist")


def test_resolve(fresolver):
    # We start from an empty path list
    fresolver.clear()

    # We add a path
    base = os.path.join(os.getenv("ERADIATE_DIR"), "eradiate/util/")
    fresolver.append(base)

    # We expect the test script to find its own location
    assert str(fresolver.resolve("tests/")) == os.path.join(base, "tests")
    assert str(fresolver.resolve("tests/test_fresolver.py")) == \
           os.path.join(base, "tests/test_fresolver.py")

    # We expect to get the original path if the file is not found
    assert str(fresolver.resolve("some_file.py")) == "some_file.py"
