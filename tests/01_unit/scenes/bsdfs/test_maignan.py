import mitsuba as mi
import pytest

from eradiate.scenes.bsdfs import MaignanBSDF
from eradiate.scenes.spectra import UniformSpectrum
from eradiate.test_tools.types import check_scene_element

MAIGNAN_PARAMS = ["C", "ndvi", "refr_re", "refr_im", "ext_ior"]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"C": 7.0, "ndvi": 0.5, "refr_re": 1.3, "refr_im": 0.01, "ext_ior": 1.1}],
    ids=["noargs", "args"],
)
def test_maignan_construct(modes_all, kwargs):
    # Default constructor
    bsdf = MaignanBSDF(**kwargs)

    for param in MAIGNAN_PARAMS:
        spectrum = getattr(bsdf, param)
        assert isinstance(spectrum, UniformSpectrum), (
            f"Parameter '{param}' is not a UniformSpectrum"
        )


def test_maignan_kernel_dict(modes_all_double):
    bsdf = MaignanBSDF(C=7.0, ndvi=0.5, refr_re=1.3, refr_im=0.01, ext_ior=1.1)

    mi_wrapper = check_scene_element(bsdf, mi.BSDF)
    for param, expected in {
        "C": 7.0,
        "ndvi": 0.5,
        "refr_re": 1.3,
        "refr_im": 0.01,
        "ext_ior": 1.1,
    }.items():
        value = mi_wrapper.parameters[f"{param}.value"]
        assert value == expected, (
            f"Parameter '{param}': expected {expected}, got {value}"
        )
