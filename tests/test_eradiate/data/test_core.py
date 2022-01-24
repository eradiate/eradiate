from eradiate.data._core import registry_from_file, registry_to_file


def test_registry_from_file(tmpdir):
    filename = tmpdir / "registry.txt"
    with open(filename, "w") as f:
        f.write(
            """
            # Registry test file
            foo1.nc
            foo2.nc sha256:bar
            foo3.nc sha256:bar https://www.foo.bar
            # Another comment line

            # Empty line above
            # By the way, the indentation will be stripped upon reading
            """
        )

    registry = registry_from_file(filename)
    assert registry == {
        "foo1.nc": "",
        "foo2.nc": "sha256:bar",
        "foo3.nc": "sha256:bar https://www.foo.bar",
    }


def test_registry_to_file(tmpdir):
    filename = tmpdir / "registry.txt"
    registry = {
        "foo1.nc": "",
        "foo2.nc": "sha256:bar",
        "foo3.nc": "sha256:bar https://www.foo.bar",
    }

    registry_to_file(registry, filename)

    with open(filename, "r") as f:
        s = f.read()

    assert s == "foo1.nc\nfoo2.nc sha256:bar\nfoo3.nc sha256:bar https://www.foo.bar\n"
