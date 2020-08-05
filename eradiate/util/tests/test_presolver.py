import os
from pathlib import Path

import pytest

from eradiate.util.presolver import PathResolver


@pytest.fixture()
def presolver():
    # This small fixture will avoid side effects
    PathResolver().reset()
    yield PathResolver()
    # Teardown: Reset file resolver again
    PathResolver().reset()


def test_contains(presolver):
    # Assuming default path, we check that the resolver is correctly initialised
    assert presolver.contains(os.getcwd())
    assert presolver.contains(".")

    # We check that nonexisting elements are not found
    assert not presolver.contains("/")


def test_remove(presolver):
    # Remove from item
    presolver.remove(os.getcwd())
    assert len(presolver) == 1

    # Remove from index
    presolver.remove(0)
    assert len(presolver) == 0

    # We expect a removal attempt with a nonexisting item to raise
    with pytest.raises(ValueError):
        presolver.remove(os.getcwd())


def test_prepend_append(presolver):
    # We start from an empty path list
    presolver.clear()

    # Test prepend
    presolver.prepend(".")
    assert presolver.contains(Path.cwd())

    # We can't prepend a nonexisting directory
    with pytest.raises(ValueError):
        presolver.prepend("some/random/path/which/doesnt/exist")

    # Test append
    presolver.append("/")
    assert presolver[1] == Path("/")

    # We can't append a nonexisting directory
    with pytest.raises(ValueError):
        presolver.append("some/random/path/which/doesnt/exist")


def test_resolve(presolver):
    # We start from an empty path list
    presolver.clear()

    # We add a path
    base = os.path.join(os.getenv("ERADIATE_DIR"), "eradiate/util/")
    presolver.append(base)

    # We expect the test script to find its own location
    assert str(presolver.resolve("tests/")) == os.path.join(base, "tests")
    assert str(presolver.resolve("tests/test_presolver.py")) == \
           os.path.join(base, "tests/test_presolver.py")

    # We expect to get the original path if the file is not found
    assert str(presolver.resolve("some_file.py")) == "some_file.py"
