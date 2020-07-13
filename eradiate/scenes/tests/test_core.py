import importlib

import attr
import pytest

import eradiate.kernel
from eradiate.scenes.core import Factory, KernelDict, SceneHelper
from eradiate.util.collections import frozendict
from eradiate.util.exceptions import KernelVariantError


def test_kernel_dict():
    # Check that object creation is possible only if a variant is set
    importlib.reload(eradiate.kernel)  # Required to ensure that any variant set by another test is unset
    with pytest.raises(KernelVariantError):
        KernelDict()
    eradiate.kernel.set_variant("scalar_mono")

    # Check that empty factory behaves as intended
    kernel_dict = KernelDict.empty()
    assert kernel_dict == {"type": "scene"}

    # Check that variant attribute is set properly
    kernel_dict = KernelDict({})
    assert kernel_dict.variant == "scalar_mono"

    # Check that check method raises upon missing scene type
    kernel_dict = KernelDict({})
    with pytest.raises(ValueError):
        kernel_dict.check()

    # Check that normalize method works as intended
    kernel_dict.normalize()
    kernel_dict.check()

    # Check that check method raises if dict and set variants are incompatible
    eradiate.kernel.set_variant("scalar_mono_double")
    with pytest.raises(KernelVariantError):
        kernel_dict.check()

    # Check that load method works
    eradiate.kernel.set_variant("scalar_mono_double")
    kernel_dict = KernelDict({"type": "scene", "shape": {"type": "sphere"}})
    assert kernel_dict.load() is not None

    # Check that add method works as intended with dicts
    kernel_dict = KernelDict.empty()
    kernel_dict.add({"shape": {"type": "sphere"}})
    assert kernel_dict == {"type": "scene", "shape": {"type": "sphere"}}


@attr.s
class TinyDirectional(SceneHelper):
    CONFIG_SCHEMA = frozendict({
        "direction": {
            "type": "list",
            "items": [{"type": "number"}] * 3,
            "default": [0, 0, -1]
        },
        "irradiance": {
            "type": "number",
            "default": 1.0
        }
    })

    id = attr.ib(default="illumination")

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "directional",
                "irradiance": self.config["irradiance"]
            }
        }


def test_scene_helper(variant_scalar_mono):
    # Default constructor (check if defaults are applied as intended)
    d = TinyDirectional()
    assert d.config == {"direction": [0, 0, -1], "irradiance": 1.0}

    # Check that undesired params raise as intended
    with pytest.raises(ValueError):
        d = TinyDirectional({
            "direction": [0, 0, -1],
            "irradiance": 1.0,
            "unexpected_param": 0
        })

    # Check that created scene can be instantiated by the kernel
    kernel_dict = KernelDict.empty()
    kernel_dict.add(d)
    assert kernel_dict.load() is not None

    # Construct using from_dict factory
    assert d == TinyDirectional.from_dict({})


def test_factory():
    factory = Factory()

    # We expect that correct object specification will yield an object
    assert factory.create({"type": "directional", "zenith": 45.}) is not None

    # We expect that incorrect object specification will raise
    # (here, the 'direction' field is not part of the expected parameters)
    with pytest.raises(ValueError):
        factory.create({"type": "directional", "direction": [0, -1, -1]})

    # We expect that an empty dict will raise
    with pytest.raises(KeyError):
        factory.create({})

    # We expect that an unregistered 'type' will raise
    with pytest.raises(ValueError):
        factory.create({"type": "dzeiaticional"})
