from pprint import pprint

import pytest

from eradiate.scenes.factory import Factory


def test_factory():
    factory = Factory()
    pprint(factory.table)
    assert factory.create({"type": "directional", "direction": [0, -1, -1]}) is not None

    with pytest.raises(KeyError):
        factory.create({})

    with pytest.raises(ValueError):
        factory.create({"type": "dzeiaticional"})