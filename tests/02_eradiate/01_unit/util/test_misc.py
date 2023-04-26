import os
from pathlib import Path

import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.util.misc import (
    Singleton,
    cache_by_id,
    camel_to_snake,
    deduplicate,
    fullname,
    is_vector3,
    natsorted,
)


def test_singleton():
    class MySingleton(metaclass=Singleton):
        pass

    my_singleton1 = MySingleton()
    my_singleton2 = MySingleton()
    assert my_singleton1 is my_singleton2


@pytest.mark.parametrize(
    "value, expected",
    [
        ("aaa", False),
        ([0, 1], False),
        ([0, 2], False),
        ([0, 1, 2], True),
        ([0.0, 1, 2], True),
        ([0, 1, "2"], False),
        (np.array([0, 1, 2]), True),
        (np.array([0, 1]), False),
        (np.array(["0", 1, 2]), False),
        (ureg.Quantity([0, 1, 2], "m"), True),
    ],
)
def test_is_vector3(value, expected):
    result = is_vector3(value)
    assert result == expected


def test_natsort():
    assert natsorted(["10", "1.2", "9"]) == ["1.2", "9", "10"]
    assert natsorted(["1.2", "a1", "9"]) == ["1.2", "9", "a1"]


def test_deduplicate():
    assert deduplicate([1, 1, 2, 3], preserve_order=True) == [1, 2, 3]
    assert deduplicate([2, 1, 3, 1], preserve_order=True) == [2, 1, 3]
    assert deduplicate([2, 1, 3, 1], preserve_order=False) == [1, 2, 3]
    # Note: this latter test may not be reproducible, we might have to change it


def test_camel_to_snake():
    assert camel_to_snake("SomeKindOfThing") == "some_kind_of_thing"


def test_fullname(mode_mono):
    # Functions
    assert fullname(os.path.join) == "posixpath.join"
    assert (
        fullname(eradiate.kernel.gridvolume.write_binary_grid3d)
        == "eradiate.kernel.gridvolume.write_binary_grid3d"
    )
    assert (
        fullname(eradiate.kernel.bitmap_to_dataset)
        == "eradiate.kernel._bitmap.bitmap_to_dataset"
    )

    # Methods
    # -- Instance method from class definition
    assert fullname(Path.is_file) == "pathlib.Path.is_file"
    # -- Instance method from instance
    path = Path()
    assert fullname(path.is_file) == "pathlib.Path.is_file"
    # -- Class method from class definition
    assert (
        fullname(eradiate.spectral.SpectralIndex.new)
        == "eradiate.spectral.index.SpectralIndex.new"
    )
    assert (
        fullname(eradiate.spectral.SpectralIndex.new().new)
        == "eradiate.spectral.index.SpectralIndex.new"
    )

    # Classes
    assert fullname(Path) == "pathlib.Path"
    assert (
        fullname(eradiate.scenes.spectra.Spectrum)
        == "eradiate.scenes.spectra._core.Spectrum"
    )


def test_cache_by_id(capsys):
    # Function
    @cache_by_id
    def f(x, y):
        print("Calling f")
        return x, y

    assert f(1, 1) == (1, 1)
    captured = capsys.readouterr()
    assert captured.out == "Calling f\n"
    assert f(1, 1) == (1, 1)
    captured = capsys.readouterr()
    assert captured.out == ""

    assert f(1, 2) == (1, 2)
    captured = capsys.readouterr()
    assert captured.out == "Calling f\n"
    assert f(1, 2) == (1, 2)
    captured = capsys.readouterr()
    assert captured.out == ""

    # Class
    class MyClass:
        @cache_by_id
        def f(self, x, y):
            print("Calling f")
            return x, y

    obj = MyClass()

    assert obj.f(1, 1) == (1, 1)
    captured = capsys.readouterr()
    assert captured.out == "Calling f\n"
    assert obj.f(1, 1) == (1, 1)
    captured = capsys.readouterr()
    assert captured.out == ""

    assert obj.f(1, 2) == (1, 2)
    captured = capsys.readouterr()
    assert captured.out == "Calling f\n"
    assert obj.f(1, 2) == (1, 2)
    captured = capsys.readouterr()
    assert captured.out == ""
