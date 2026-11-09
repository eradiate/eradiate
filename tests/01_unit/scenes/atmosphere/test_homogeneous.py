import drjit as dr
import pytest

from eradiate import unit_registry as ureg
from eradiate.scenes.atmosphere import Atmosphere, HomogeneousComponent
from eradiate.scenes.core import traverse
from eradiate.spectral.index import SpectralIndex
from eradiate.testing.types import check_scene_element


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
def test_homogeneous_component_construct(modes_all_double, kwargs):
    geometry = kwargs.pop("geometry", "plane_parallel")
    component = HomogeneousComponent(**kwargs)
    check_scene_element(Atmosphere(geometry=geometry, components=[component]))


def test_homogeneous_component_params(mode_mono):
    component = HomogeneousComponent(
        phase={
            "type": "hg",
            "g": {
                "type": "interpolated",
                "wavelengths": [400, 700],
                "values": [0.0, 0.8],
            },
        }
    )
    atmosphere = Atmosphere(components=[component])

    _, umap_template = traverse(atmosphere)
    mi_wrapper = check_scene_element(atmosphere)

    # Phase function parameters are exposed at highest level
    assert "phase_atmosphere.g" in umap_template
    assert "phase_atmosphere.g" in mi_wrapper.parameters
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
                "atmosphere_shape_type_name": "AnalyticalCube",
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
def test_homogeneous_component_geometry(mode_mono, kwargs, expected):
    atmosphere = Atmosphere(components=[HomogeneousComponent()], **kwargs)
    mi_wrapper = check_scene_element(atmosphere)

    # Check scene shape type
    mi_shape_type_name = mi_wrapper.obj.shapes()[0].class_name()
    assert mi_shape_type_name == expected["atmosphere_shape_type_name"]

    # Check scene bounding box
    bbmin = mi_wrapper.obj.bbox().min
    bbmax = mi_wrapper.obj.bbox().max
    assert dr.allclose(bbmin, expected["bbox_min"])
    assert dr.allclose(bbmax, expected["bbox_max"])


def test_homogeneous_component_eval(mode_mono):
    component = HomogeneousComponent(sigma_s=3e-5, sigma_a=1e-5)
    si = SpectralIndex.new()

    # The grid argument is ignored, values are scalar
    assert component.eval_sigma_t(si, grid=None).m_as("1/m") == pytest.approx(4e-5)
    assert component.eval_albedo(si).m_as("") == pytest.approx(0.75)
