import drjit as dr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.atmosphere import HomogeneousAtmosphere
from eradiate.scenes.core import traverse
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"phase": {"type": "isotropic"}},
        {"phase": {"type": "hg"}},
        {"bottom": 1.0 * ureg.km, "top": 8.0 * ureg.km},
        {"geometry": "spherical_shell"},
    ],
    ids=[
        "noargs",
        "phase_isotropic",
        "phase_hg",
        "bottom_top",
        "geometry_spherical_shell",
    ],
)
def test_homogeneous_atmosphere_construct(modes_all_double, kwargs):
    atmosphere = HomogeneousAtmosphere(**kwargs)
    check_scene_element(atmosphere)


def test_homogeneous_atmosphere_params(mode_mono):
    atmosphere = HomogeneousAtmosphere(
        phase={
            "type": "hg",
            "g": {
                "type": "interpolated",
                "wavelengths": [400, 700],
                "values": [0.0, 0.8],
            },
        }
    )

    # Phase function parameters are exposed at highest level
    _, params = traverse(atmosphere)
    assert "medium_atmosphere.phase_function.g" in params

    _, mi_params = check_scene_element(atmosphere)
    assert "medium_atmosphere.phase_function.g" in mi_params


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {
                "geometry": {"type": "plane_parallel", "width": 1.0 * ureg.km},
                "bottom": 0.0 * ureg.km,
                "top": 1.0 * ureg.km,
            },
            {
                "bbox_min": [-500.0, -500.0, -10.0],
                "bbox_max": [500.0, 500.0, 1000.0],
                "atmosphere_shape_type_name": "Cube",
            },
        ),
        (
            {
                "geometry": {"type": "spherical_shell", "planet_radius": 1.0 * ureg.km},
                "bottom": 0.0 * ureg.km,
                "top": 1.0 * ureg.km,
            },
            {
                "bbox_min": [-2000.0, -2000.0, -2000.0],
                "bbox_max": [2000.0, 2000.0, 2000.0],
                "atmosphere_shape_type_name": "Sphere",
            },
        ),
    ],
    ids=[
        "plane_parallel",
        "spherical_shell",
    ],
)
def test_homogeneous_atmosphere_geometry(mode_mono, kwargs, expected):
    atmosphere = HomogeneousAtmosphere(**kwargs)
    mi_obj, mi_params = check_scene_element(atmosphere)

    # Check scene shape type
    mi_shape_type_name = mi_obj.shapes()[0].class_().name()
    assert mi_shape_type_name == expected["atmosphere_shape_type_name"]

    # Check scene bounding box
    assert dr.allclose(mi_obj.bbox().min, expected["bbox_min"])
    assert dr.allclose(mi_obj.bbox().max, expected["bbox_max"])
