import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import BSDF, LambertianBSDF
from eradiate.scenes.core import NodeSceneElement
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.test_tools.types import check_node_scene_element, check_type


def test_lambertian_type():
    check_type(
        LambertianBSDF,
        expected_mro=[BSDF, NodeSceneElement],
        expected_slots=[],
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"reflectance": 0.5}],
    ids=["noargs", "args"],
)
def test_lambertian_construct(modes_all, kwargs):
    # Default constructor
    bsdf = LambertianBSDF(**kwargs)
    assert isinstance(bsdf.reflectance, UniformSpectrum)


def test_lambertian_kernel_dict(modes_all_double):
    bsdf = LambertianBSDF(reflectance=0.75)

    mi_obj, mi_params = check_node_scene_element(bsdf, mi.BSDF)
    assert mi_params["reflectance.value"] == 0.75
