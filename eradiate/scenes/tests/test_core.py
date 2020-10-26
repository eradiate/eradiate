import importlib

import attr
import numpy as np
import pytest

import eradiate.kernel
from eradiate.scenes.core import SceneElementFactory, KernelDict, SceneElement
from eradiate.util.attrs import attrib, attrib_units, validator_has_len, validator_is_number, validator_is_positive
from eradiate.util.exceptions import KernelVariantError
from eradiate.util.units import ureg


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
class TinyDirectional(SceneElement):
    id = attrib(
        default="illumination",
        validator=attr.validators.instance_of(str),
    )

    direction, direction_units = attrib(
        default=[0, 0, -1],
        validator=validator_has_len(3),
        units_compatible=ureg.m,
        units_default=ureg.m
    )

    irradiance = attr.ib(
        default=1.0,
        validator=[validator_is_number, validator_is_positive],
    )

    def kernel_dict(self, **kwargs):
        direction = self.direction * self.direction_units

        return {
            self.id: {
                "type": "directional",
                "irradiance": self.irradiance,
                "direction": direction.to("m").magnitude
            }
        }


def test_scene_element(mode_mono):
    # Default constructor (check if defaults are applied as intended)
    d = TinyDirectional()
    assert attr.asdict(d) == {
        "id": "illumination",
        "direction": [0, 0, -1],
        "direction_units": ureg.m,
        "irradiance": 1.0
    }

    # Check that constructor from params works as intended
    assert TinyDirectional(
        direction=[0, 0, -100],
        direction_units=ureg.cm,
        irradiance=1.0
    ) is not None

    # Check that unit handling is appropriately performed
    d = TinyDirectional(
        direction=ureg.Quantity([0, 0, -100], ureg.cm),
        direction_units=ureg.m
    )
    assert np.allclose(d.direction, [0, 0, -1])

    # Check that constructor from dictionary works as intended
    assert TinyDirectional(direction=[0, 0, -1], irradiance=1.0) == \
        TinyDirectional.from_dict({"direction": [0, 0, -1], "irradiance": 1.0})
    assert TinyDirectional() == TinyDirectional.from_dict({})

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


def test_factory():
    # We expect that correct object specification will yield an object
    assert SceneElementFactory.create({"type": "directional", "zenith": 45.}) is not None

    # We expect that incorrect object specification will raise
    # (here, the 'direction' field is not part of the expected parameters)
    with pytest.raises(TypeError):
        SceneElementFactory.create({"type": "directional", "direction": [0, -1, -1]})

    # We expect that an empty dict will raise
    with pytest.raises(KeyError):
        SceneElementFactory.create({})

    # We expect that an unregistered 'type' will raise
    with pytest.raises(ValueError):
        SceneElementFactory.create({"type": "dzeiaticional"})

    # Test converter
    assert isinstance(SceneElementFactory.convert({"type": "directional", "zenith": 45.}),
                      DirectionalIllumination)
    assert isinstance(SceneElementFactory.convert(DirectionalIllumination()),
                      DirectionalIllumination)
