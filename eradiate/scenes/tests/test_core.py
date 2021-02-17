import importlib

import attr
import numpy as np
import pinttr
import pytest

import eradiate.kernel
from eradiate import unit_context_config as ucc
from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict, SceneElement
from eradiate.validators import has_len, is_number, is_positive
from eradiate.exceptions import KernelVariantError


def test_kernel_dict():
    # Check that object creation is possible only if a variant is set
    importlib.reload(
        eradiate.kernel)  # Required to ensure that any variant set by another test is unset
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


def test_scene_element(mode_mono):
    @attr.s
    class TinyDirectional(SceneElement):
        id = attr.ib(
            default="illumination",
            validator=attr.validators.instance_of(str),
        )

        direction = pinttr.ib(
            default=ureg.Quantity([0, 0, -1], ureg.m),
            units=ucc.deferred("length"),
        )

        @direction.validator
        def _direction_validator(self, attribute, value):
            has_len(3)(self, attribute, value)

        irradiance = attr.ib(
            default=1.0,
            validator=[is_number, is_positive],
        )

        def kernel_dict(self, ref=True):
            return {
                self.id: {
                    "type": "directional",
                    "irradiance": self.irradiance,
                    "direction": self.direction.to("m").magnitude
                }
            }

    # Dict initialiser tests
    # -- Check if scalar + units yields proper field value
    d = TinyDirectional.from_dict({
        "direction": [0, 0, -1],
        "direction_units": "km"
    })
    assert np.allclose(d.direction, ureg.Quantity([0, 0, -1], ureg.km))
    # -- Check if scalar is attached default units as expected
    d = TinyDirectional.from_dict({
        "direction": [0, 0, -1]
    })
    assert np.allclose(d.direction, ureg.Quantity([0, 0, -1], ureg.m))
    # -- Check if quantity is attached default units as expected
    d = TinyDirectional.from_dict({
        "direction": ureg.Quantity([0, 0, -1], "km")
    })
    assert np.allclose(d.direction, ureg.Quantity([0, 0, -1], ureg.km))
    # -- Check if the unit field can be used to force conversion of quantity
    d = TinyDirectional.from_dict({
        "direction": ureg.Quantity([0, 0, -1], "km"),
        "direction_units": "m"
    })
    print(d)
    assert np.allclose(d.direction, ureg.Quantity([0, 0, -1], ureg.km))
    assert d.direction.units == ureg.m

    # Setter tests
    d = TinyDirectional()
    # -- Check that assigned non-quantity gets wrapped into quantity
    d.direction = [0, 0, -1]
    assert d.direction.units == ureg.m
    # -- Check that default units get applied when overridden
    with ucc.override({"length": "km"}):
        d.direction = [0, 0, -1]
    assert np.allclose(d.direction, [0, 0, -1000] * ureg.m)
    assert d.direction.units == ureg.km
    # -- Setting with incompatible units should raise
    with pytest.raises(pinttr.exceptions.UnitsError):
        d.direction = [0, 0, -1] * ureg.s

    # Check that created scene can be instantiated by the kernel
    kernel_dict = KernelDict.empty()
    kernel_dict.add(d)
    assert kernel_dict.load() is not None

    # Check that undesired parameters raise
    with pytest.raises(TypeError):
        TinyDirectional.from_dict({
            "direction": [0, 0, -1],
            "irradiance": 1.0,
            "unexpected_param": 0
        })
