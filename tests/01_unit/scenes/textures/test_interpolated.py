import mitsuba as mi
import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.scenes.textures import InterpolatedTexture
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "wavelengths, data, expected",
    [
        (None, None, TypeError),
        ([500.0, 600.0], None, TypeError),
        (None, [[[0.0, 1.0]]], TypeError),
        ([500.0, 600.0], [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], ValueError),
        ([500.0, 600.0], [[[0.0], [1.0]], [[0.0], [1.0]]], ValueError),
        ([500.0, 400.0], [[[0.0, 1.0]]], ValueError),
        (
            [400.0, 500.0],
            [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
            InterpolatedTexture(
                wavelengths=[400.0, 500.0],
                data=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
            ),
        ),
    ],
    ids=[
        "no_args",
        "missing_args_1",
        "missing_args_2",
        "values_shape_1",
        "values_shape_2",
        "wavelength_monotonicity",
        "valid",
    ],
)
def test_interpolated_texture_construct(modes_all, wavelengths, data, expected):
    if isinstance(expected, InterpolatedTexture):
        t = InterpolatedTexture(wavelengths=wavelengths, data=data)
        assert np.all(t.data == expected.data)
        assert np.all(t.wavelengths == expected.wavelengths)

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            if not wavelengths:
                InterpolatedTexture(data=data)
            elif not data:
                InterpolatedTexture(wavelengths=wavelengths)
            else:
                InterpolatedTexture(wavelengths=wavelengths, data=data)

    else:
        raise RuntimeError


def test_interpolated_texture_eval(modes_all):
    if eradiate.mode().is_mono:
        si = SpectralIndex.new(w=550.0 * ureg.nm)
        expected = 0.5

    elif eradiate.mode().is_ckd:
        si = SpectralIndex.new(w=550.0 * ureg.nm, g=0)
        expected = 0.5

    else:
        raise NotImplementedError

    t = InterpolatedTexture(
        wavelengths=[500, 600],
        data=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
    )

    assert np.allclose(t.eval(si), expected)


def test_interpolated_texture_kernel_dict(modes_all):
    t = InterpolatedTexture(
        wavelengths=[500, 600],
        data=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
    )
    mi_wrapper = check_scene_element(t, mi.Texture)

    assert np.allclose(mi_wrapper.parameters["data"], 0.5)
