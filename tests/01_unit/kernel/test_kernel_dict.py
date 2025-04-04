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
    kdict = KernelDict({"foo": {"bar": 0}, "bar": {"baz": {"qux": 0}}})
    assert kdict.data == {"foo.bar": 0, "bar.baz.qux": 0}


def test_kernel_dict_setitem():
    kdict = KernelDict()
    kdict["foo"] = {"bar": 0}
    kdict["bar"] = {"baz": {"qux": 0}}
    kdict["baz"] = 0
    assert kdict.data == {"foo.bar": 0, "bar.baz.qux": 0, "baz": 0}


def test_kernel_dict_render():
    kdict = KernelDict(
        {
            "foo.bar": 0,
            "bar": DictParameter(lambda ctx: ctx),
            "baz": DictParameter(lambda ctx: ctx),
        }
    )

    assert kdict.render(ctx=1, nested=True) == {
        "foo": {"bar": 0},
        "bar": 1,
        "baz": 1,
    }

    assert kdict.render(ctx=1, nested=False) == {"foo.bar": 0, "bar": 1, "baz": 1}


def test_scene_parameter_map_render():
    kpmap = KernelSceneParameterMap(
        {
            "foo": SceneParameter(func=lambda ctx: ctx, tracks="foo"),
            "bar": SceneParameter(
                func=lambda ctx: ctx,
                flags=KernelSceneParameterFlags.GEOMETRIC,
                tracks="bar",
            ),
            "baz": SceneParameter(
                func=lambda ctx: ctx,
                flags=KernelSceneParameterFlags.SPECTRAL,
                tracks="baz",
            ),
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
