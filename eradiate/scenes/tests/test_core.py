import importlib

import mitsuba
import pytest

from eradiate.exceptions import KernelVariantError
from eradiate.scenes.core import KernelDict


def test_kernel_dict():
    # Object creation is possible only if a variant is set
    importlib.reload(
        mitsuba
    )  # Required to ensure that any variant set by another test is unset
    with pytest.raises(KernelVariantError):
        KernelDict()
    mitsuba.set_variant("scalar_mono")

    # Constructor class method initialises an empty kernel scene dict
    kernel_dict = KernelDict.new()
    assert kernel_dict == {"type": "scene"}

    # variant attribute is set properly
    kernel_dict = KernelDict({})
    assert kernel_dict.variant == "scalar_mono"

    # Check method raises upon missing scene type
    kernel_dict = KernelDict({})
    with pytest.raises(ValueError):
        kernel_dict.check()

    # Check method raises if dict and set variants are incompatible
    mitsuba.set_variant("scalar_mono_double")
    with pytest.raises(KernelVariantError):
        kernel_dict.check()

    # Load method returns a kernel object
    mitsuba.set_variant("scalar_mono_double")
    kernel_dict = KernelDict({"type": "scene", "shape": {"type": "sphere"}})
    assert kernel_dict.load() is not None

    # Add method merges dicts
    kernel_dict = KernelDict.new()
    kernel_dict.add({"shape": {"type": "sphere"}})
    assert kernel_dict == {"type": "scene", "shape": {"type": "sphere"}}
