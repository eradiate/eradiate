import importlib

import mitsuba
import numpy as np
import pytest

from eradiate.exceptions import KernelVariantError
from eradiate.scenes.core import KernelDict


def test_kernel_dict_construct():
    # Object creation is possible only if a variant is set
    importlib.reload(
        mitsuba
    )  # Required to ensure that any variant set by another test is unset
    with pytest.raises(KernelVariantError):
        KernelDict()
    mitsuba.set_variant("scalar_mono")

    # variant attribute is set properly
    kernel_dict = KernelDict({})
    assert kernel_dict.variant == "scalar_mono"


def test_kernel_dict_check(mode_mono):
    # Check method raises upon missing scene type
    kernel_dict = KernelDict({})
    with pytest.raises(ValueError):
        kernel_dict.check()

    # Check method raises if dict and set variants are incompatible
    mitsuba.set_variant("scalar_mono_double")
    with pytest.raises(KernelVariantError):
        kernel_dict.check()


def test_kernel_dict_load(mode_mono):
    # Load method returns a kernel object
    from mitsuba.render import Scene, Shape

    kernel_dict = KernelDict({"type": "scene", "shape": {"type": "sphere"}})
    assert isinstance(kernel_dict.load(), Scene)

    # Also works if "type" is missing
    kernel_dict = KernelDict({"shape": {"type": "sphere"}})
    with pytest.warns(UserWarning):
        obj = kernel_dict.load(strip=False)
    assert isinstance(obj, Scene)

    # Setting strip to True instantiates a Shape directly...
    kernel_dict = KernelDict({"shape": {"type": "sphere"}})
    assert isinstance(kernel_dict.load(strip=True), Shape)

    # ... but not if the dict has two entries
    kernel_dict = KernelDict(
        {
            "shape_1": {"type": "sphere"},
            "shape_2": {"type": "sphere"},
        }
    )
    with pytest.warns(UserWarning):
        obj = kernel_dict.load(strip=True)
    assert isinstance(obj, Scene)


def test_kernel_dict_post_load(mode_mono):
    from mitsuba.python.util import traverse

    kernel_dict = KernelDict(
        data={
            "type": "directional",
            "irradiance": {
                "type": "irregular",
                "wavelengths": "400, 500",
                "values": "1, 1",
            },
        },
        post_load={
            "irradiance.wavelengths": np.array([400.0, 500.0, 600.0]),
            "irradiance.values": np.array([0.0, 1.0, 2.0]),
        },
    )

    # Without post-load update, buffers are initialised as in data
    obj = kernel_dict.load(post_load_update=False)
    params = traverse(obj)
    assert params["irradiance.wavelengths"] == np.array([400.0, 500.0])
    assert params["irradiance.values"] == np.array([1.0, 1.0])

    # Without post-load update, buffers are initialised as in post_load
    obj = kernel_dict.load(post_load_update=True)
    params = traverse(obj)
    assert params["irradiance.wavelengths"] == np.array([400.0, 500.0, 600.0])
    assert params["irradiance.values"] == np.array([0.0, 1.0, 2.0])
