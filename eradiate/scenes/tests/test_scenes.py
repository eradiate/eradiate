import importlib

import pytest

import eradiate.kernel
from eradiate.scenes import SceneDict
from eradiate.util.exceptions import KernelVariantError


def test_scene_dict():
    # Check that object creation is possible only if a variant is set
    importlib.reload(eradiate.kernel)  # Required to ensure that any variant set by another test is unset
    with pytest.raises(KernelVariantError):
        SceneDict()
    eradiate.kernel.set_variant("scalar_mono")

    # Check that empty factory behaves as intended
    scene_dict = SceneDict.empty()
    assert scene_dict == {"type": "scene"}

    # Check that variant attribute is set properly
    scene_dict = SceneDict({})
    assert scene_dict.variant == "scalar_mono"

    # Check that check method raises upon missing scene type
    scene_dict = SceneDict({})
    with pytest.raises(ValueError):
        scene_dict.check()

    # Check that normalize method works as intended
    scene_dict.normalize()
    scene_dict.check()

    # Check that check method raises if dict and set variants are incompatible
    eradiate.kernel.set_variant("scalar_mono_double")
    with pytest.raises(KernelVariantError):
        scene_dict.check()

    # Check that load method works
    eradiate.kernel.set_variant("scalar_mono_double")
    scene_dict = SceneDict({"type": "scene", "shape": {"type": "sphere"}})
    assert scene_dict.load() is not None

    # Check that add method works as intended with dicts
    scene_dict = SceneDict.empty()
    scene_dict.add({"shape": {"type": "sphere"}})
    assert scene_dict == {"type": "scene", "shape": {"type": "sphere"}}
