import re

import pytest

from eradiate.kernel import (
    DictParameter,
    KernelDict,
    KernelSceneParameterFlags,
    KernelSceneParameterMap,
    SceneParameter,
)


def test_kernel_dict_init():
    template = KernelDict({"foo": {"bar": 0}, "bar": {"baz": {"qux": 0}}})
    assert template.data == {"foo.bar": 0, "bar.baz.qux": 0}


def test_kernel_dict_setitem():
    template = KernelDict()
    template["foo"] = {"bar": 0}
    template["bar"] = {"baz": {"qux": 0}}
    template["baz"] = 0
    assert template.data == {"foo.bar": 0, "bar.baz.qux": 0, "baz": 0}


def test_kernel_dict_render():
    template = KernelDict(
        {
            "foo.bar": 0,
            "bar": DictParameter(lambda ctx: ctx),
            "baz": DictParameter(lambda ctx: ctx),
        }
    )

    assert template.render(ctx=1, nested=True) == {
        "foo": {"bar": 0},
        "bar": 1,
        "baz": 1,
    }

    assert template.render(ctx=1, nested=False) == {"foo.bar": 0, "bar": 1, "baz": 1}


def test_scene_parameter_map_render():
    kpmap = KernelSceneParameterMap(
        {
            "foo": 0,
            "bar": SceneParameter(lambda ctx: ctx, KernelSceneParameterFlags.GEOMETRIC),
            "baz": SceneParameter(lambda ctx: ctx, KernelSceneParameterFlags.SPECTRAL),
        }
    )

    # If no flags are passed, all params are rendered
    result = kpmap.render(ctx=1, flags=KernelSceneParameterFlags.ALL)
    assert result["bar"] == 1 and result["baz"] == 1

    # If a flag is passed, only the corresponding params are rendered
    with pytest.raises(ValueError, match=re.escape("Unevaluated parameters: ['bar']")):
        kpmap.render(ctx=1, flags=KernelSceneParameterFlags.SPECTRAL, drop=False)

    # If drop is set to True, unused parameters are dropped
    result = kpmap.render(ctx=1, flags=KernelSceneParameterFlags.SPECTRAL, drop=True)
    assert result["baz"] == 1
    assert "bar" not in result
