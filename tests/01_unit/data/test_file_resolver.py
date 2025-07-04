import pytest

from eradiate.data import FileResolver


def test_file_resolver_construct(tmpdir):
    # File resolver can be initialized empty
    assert FileResolver()

    # Can be initialized with a directory that exists
    assert FileResolver([tmpdir])

    # Paths that do not exist raise
    with pytest.raises(NotADirectoryError):
        FileResolver(["does/not/exist"])


def test_file_resolver_append(tmpdir):
    path_a = tmpdir / "a"
    path_a.mkdir()
    path_b = tmpdir / "b"
    path_b.mkdir()

    fresolver = FileResolver()
    fresolver.append(path_a)
    fresolver.append(path_b)
    assert fresolver.paths == [path_a, path_b]

    fresolver.append(path_a)
    assert fresolver.paths == [path_a, path_b]

    fresolver.append(path_a, avoid_duplicates=False)
    assert fresolver.paths == [path_a, path_b, path_a]


def test_file_resolver_prepend(tmpdir):
    path_a = tmpdir / "a"
    path_a.mkdir()
    path_b = tmpdir / "b"
    path_b.mkdir()

    fresolver = FileResolver()
    fresolver.prepend(path_a)
    fresolver.prepend(path_b)
    assert fresolver.paths == [path_b, path_a]

    fresolver.prepend(path_a)
    assert fresolver.paths == [path_b, path_a]

    fresolver.prepend(path_b, avoid_duplicates=False)
    assert fresolver.paths == [path_b, path_b, path_a]


def test_file_resolver_clear(tmpdir):
    fresolver = FileResolver([tmpdir])
    fresolver.clear()
    assert fresolver.paths == []


def test_file_resolver_resolve(tmpdir):
    fresolver = FileResolver([tmpdir])

    with open(tmpdir / "foo.txt", "w") as f:
        f.write("Hello world")

    # Files that exist are resolved to absolute paths
    fname = fresolver.resolve("foo.txt")
    assert fname.is_absolute()
    assert fname.parent == tmpdir

    # Files that do not exist are not resolved
    fname = fresolver.resolve("bar.txt")
    assert not fname.is_absolute()

    # In strict mode, files that do not exist raise
    with pytest.raises(FileNotFoundError):
        fresolver.resolve("bar.txt", strict=True)
