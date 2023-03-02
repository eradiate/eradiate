import mitsuba as mi
import numpy as np
import pytest

from eradiate.scenes.bsdfs import LambertianBSDF, OpacityMaskBSDF, bsdf_factory
from eradiate.test_tools.types import check_scene_element


def test_opacity_mask_construct(modes_all):
    assert OpacityMaskBSDF(
        opacity_bitmap=mi.Bitmap(np.ones((3, 3))),
        uv_trafo=mi.ScalarTransform4f.scale(2),
        nested_bsdf=LambertianBSDF(),
    )


def test_opacity_mask_construct_dict(modes_all):
    assert bsdf_factory.convert(
        {
            "type": "opacity_mask",
            "opacity_bitmap": mi.Bitmap(np.ones((3, 3))),
            "uv_trafo": mi.ScalarTransform4f.scale(2),
            "nested_bsdf": {
                "type": "lambertian",
                "reflectance": 0.5
            }
        }
    )


def test_opacity_mask_kernel_dict(modes_all_double):
    bsdf = OpacityMaskBSDF(
        opacity_bitmap=mi.Bitmap(np.ones((3, 3))),
        uv_trafo=mi.ScalarTransform4f.scale(2),
        nested_bsdf=LambertianBSDF(),
    )
    mi_wrapper = check_scene_element(bsdf, mi.BSDF)
    assert "nested_bsdf.reflectance.value" in mi_wrapper.parameters.keys()
