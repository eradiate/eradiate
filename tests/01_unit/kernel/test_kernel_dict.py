import re

import pytest

from eradiate.kernel import (
    InitParameter,
    KernelDictTemplate,
    UpdateMapTemplate,
    UpdateParameter,
)


def test_kernel_dict_template_render():
    template = KernelDictTemplate(
        {
            "foo.bar": 0,
            "bar": InitParameter(lambda ctx: ctx),
            "baz": InitParameter(lambda ctx: ctx),
        }
    )

    assert template.render(ctx=1, nested=True) == {
        "foo": {"bar": 0},
        "bar": 1,
        "baz": 1,
    }

    assert template.render(ctx=1, nested=False) == {"foo.bar": 0, "bar": 1, "baz": 1}


def test_update_map_template_render():
    pmap = UpdateMapTemplate(
        {
            "foo": 0,
            "bar": UpdateParameter(lambda ctx: ctx, UpdateParameter.Flags.GEOMETRIC),
            "baz": UpdateParameter(lambda ctx: ctx, UpdateParameter.Flags.SPECTRAL),
        }
    )

    # If no flags are passed, all params are rendered
    result = pmap.render(ctx=1, flags=UpdateParameter.Flags.ALL)
    assert result["bar"] == 1 and result["baz"] == 1

    # If a flag is passed, only the corresponding params are rendered
    with pytest.raises(ValueError, match=re.escape("Unevaluated parameters: ['bar']")):
        pmap.render(ctx=1, flags=UpdateParameter.Flags.SPECTRAL, drop=False)

    # If drop is set to True, unused parameters are dropped
    result = pmap.render(ctx=1, flags=UpdateParameter.Flags.SPECTRAL, drop=True)
    assert result["baz"] == 1
    assert "bar" not in result


def test_update_map_template_keep_remove():
    template = UpdateMapTemplate(
        {
            "foo": 0,
            "foo.bar": 1,
            "bar": UpdateParameter(lambda ctx: ctx, UpdateParameter.Flags.GEOMETRIC),
            "baz": UpdateParameter(lambda ctx: ctx, UpdateParameter.Flags.SPECTRAL),
        }
    )

    # We can remove or keep selected parameters
    pmap = template.copy()
    pmap.remove(r"foo.*")
    assert pmap.keys() == {"bar", "baz"}

    pmap = template.copy()
    pmap.keep(r"foo.*")
    assert pmap.keys() == {"foo", "foo.bar"}
