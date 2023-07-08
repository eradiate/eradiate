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
        {"geometry": "spherical_shell"},
    ],
    ids=[
        "noargs",
        "phase_isotropic",
        "phase_hg",
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

    _, umap_template = traverse(atmosphere)
    mi_wrapper = check_scene_element(atmosphere)

    # Phase function parameters are exposed at highest level
    assert "medium_atmosphere.phase_function.g" in umap_template
    assert "medium_atmosphere.phase_function.g" in mi_wrapper.parameters.keys()
    # Volume data source parameters are exposed at highest level
    assert "medium_atmosphere.sigma_t.value.value" in umap_template
    assert "medium_atmosphere.albedo.value.value" in umap_template


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {
                "geometry": {
                    "type": "plane_parallel",
                    "width": 1.0 * ureg.km,
                    "ground_altitude": 0.0 * ureg.km,
                    "toa_altitude": 1.0 * ureg.km,
                }
            },
            {
                "bbox_min": [-500.0, -500.0, -10.0],
                "bbox_max": [500.0, 500.0, 1000.0],
                "atmosphere_shape_type_name": "Cube",
            },
        ),
        (
            {
                "geometry": {
                    "type": "spherical_shell",
                    "planet_radius": 1.0 * ureg.km,
                    "ground_altitude": 0.0 * ureg.km,
                    "toa_altitude": 1.0 * ureg.km,
                },
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
    mi_wrapper = check_scene_element(atmosphere)

    # Check scene shape type
    mi_shape_type_name = mi_wrapper.obj.shapes()[0].class_().name()
    assert mi_shape_type_name == expected["atmosphere_shape_type_name"]

    # Check scene bounding box
    bbmin = mi_wrapper.obj.bbox().min
    bbmax = mi_wrapper.obj.bbox().max
    assert dr.allclose(bbmin, expected["bbox_min"])
    assert dr.allclose(bbmax, expected["bbox_max"])
