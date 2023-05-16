import mitsuba as mi
import numpy as np
import pint
import pytest
import xarray as xr

from eradiate import KernelContext
from eradiate.constants import EARTH_RADIUS
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry
from eradiate.scenes.shapes import RectangleShape, SphereShape
from eradiate.scenes.surface import DEMSurface
from eradiate.scenes.surface._dem import (
    _generate_dem_vertices,
    _generate_face_indices,
    _mean_coordinates,
    _minmax_coordinates,
    _to_uv,
    _transform_vertices_plane_parallel,
    _transform_vertices_spherical_shell,
    _vertex_index,
    mesh_from_dem,
)
from eradiate.test_tools.types import check_scene_element
from eradiate.units import symbol
from eradiate.units import unit_registry as ureg


def make_dataarray(data: pint.Quantity, coords: dict):
    if set(coords.keys()) == {"lat", "lon"}:
        dims = ["lat", "lon"]
        coords = {
            "lat": (
                ["lat"],
                coords["lat"].magnitude,
                {"units": symbol(coords["lat"].units)},
            ),
            "lon": (
                ["lon"],
                coords["lon"].magnitude,
                {"units": symbol(coords["lon"].units)},
            ),
        }

    elif set(coords.keys()) == {"x", "y"}:
        dims = ["x", "y"]
        coords = {
            "x": (
                ["x"],
                coords["x"].magnitude,
                {"units": symbol(coords["x"].units)},
            ),
            "y": (
                ["y"],
                coords["y"].magnitude,
                {"units": symbol(coords["y"].units)},
            ),
        }

    else:
        raise ValueError

    return xr.DataArray(
        data=data.magnitude,
        dims=dims,
        coords=coords,
        attrs={"units": symbol(data.units)},
    )


@pytest.mark.parametrize(
    "kwargs, expected_limits",
    [
        (
            {
                "da": make_dataarray(
                    data=np.zeros((10, 10)) * ureg.m,
                    coords={
                        "lat": np.linspace(-0.1, 0.1, 10) * ureg.deg,
                        "lon": np.linspace(-0.1, 0.1, 10) * ureg.deg,
                    },
                ),
                "geometry": "plane_parallel",
                "planet_radius": EARTH_RADIUS,
            },
            [(-0.1, 0.1), (-0.1, 0.1)],
        ),
        (
            {
                "da": make_dataarray(
                    data=np.zeros((10, 10)) * ureg.m,
                    coords={
                        "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
                        "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
                    },
                ),
                "geometry": "spherical_shell",
                "planet_radius": EARTH_RADIUS,
            },
            [(-0.2, 0.3), (-0.4, 0.5)],
        ),
        (
            {
                "da": make_dataarray(
                    data=np.zeros((10, 10)) * ureg.m,
                    coords={
                        "x": np.linspace(-20, 20, 10) * ureg.km,
                        "y": np.linspace(-30, 30, 10) * ureg.km,
                    },
                ),
                "geometry": "plane_parallel",
                "planet_radius": EARTH_RADIUS,
            },
            [(-0.179664, 0.179664), (-0.269496, 0.269496)],
        ),
        (
            {
                "da": make_dataarray(
                    data=np.zeros((10, 10)) * ureg.m,
                    coords={
                        "x": np.linspace(-10, 20, 10) * ureg.km,
                        "y": np.linspace(-30, 40, 10) * ureg.km,
                    },
                ),
                "geometry": "spherical_shell",
                "planet_radius": EARTH_RADIUS,
            },
            [(-0.089832, 0.179664), (-0.269496, 0.359328)],
        ),
    ],
)
def test_mesh_from_dem(modes_all_double, kwargs, expected_limits):
    if (kwargs["geometry"] == "spherical_shell") and kwargs[
        "planet_radius"
    ] is not None:
        with pytest.warns():
            mesh, lat, lon = mesh_from_dem(**kwargs)
    else:
        mesh, lat, lon = mesh_from_dem(**kwargs)

    assert len(mesh.vertices) == 100

    assert np.allclose(lat.m, expected_limits[0], atol=1e-4, rtol=1e-3)
    assert np.allclose(lon.m, expected_limits[1], atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    "geometry, expected_bg, planet_radius",
    [
        (PlaneParallelGeometry(), RectangleShape, EARTH_RADIUS),
        (SphericalShellGeometry(), SphereShape, None),
    ],
)
def test_dem_surface_construct(modes_all_mono, geometry, expected_bg, planet_radius):
    mesh, lat, lon = mesh_from_dem(
        da=make_dataarray(
            data=np.zeros((10, 10)) * ureg.m,
            coords={
                "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
                "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
            },
        ),
        geometry=geometry,
        planet_radius=planet_radius,
    )

    dem = DEMSurface.from_mesh(
        id="terrain", mesh=mesh, lat=lat, lon=lon, geometry=geometry
    )
    assert isinstance(dem.shape_background, expected_bg)


def test_dem_surface_kernel_dict(mode_mono):
    geometry = SphericalShellGeometry()
    mesh, lat, lon = mesh_from_dem(
        da=make_dataarray(
            data=np.zeros((10, 10)) * ureg.m,
            coords={
                "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
                "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
            },
        ),
        geometry=geometry,
    )

    dem = DEMSurface.from_mesh(
        id="terrain", mesh=mesh, lat=lat, lon=lon, geometry=geometry
    )

    check_scene_element(dem)

    # When enclosed in a Scene, the surface can be traversed
    scene = Scene(objects={"surface": dem})
    template, params = traverse(scene)
    kernel_dict = template.render(KernelContext())
    assert isinstance(mi.load_dict(kernel_dict), mi.Scene)


def test_generate_dem_vertices(mode_mono):
    latitude_points = np.array([1, 2])
    longitude_points = np.array([3, 4])
    elevation = np.array([[1, 2], [3, 4]])

    vertices = _generate_dem_vertices(latitude_points, longitude_points, elevation)

    expected_vertices = [[1, 3, 1], [1, 4, 2], [2, 3, 3], [2, 4, 4]]

    for exp in expected_vertices:
        assert any([np.all(exp == v) for v in vertices])


def test_generate_face_indices(mode_mono):
    face_indices = _generate_face_indices(len_x=2, len_y=2)
    print(face_indices)

    expected_indices = [[3, 0, 1], [2, 0, 3]]

    for exp in expected_indices:
        assert any([np.all(exp == f) for f in face_indices])


def test_mean_coordinates(mode_mono):
    vertices = np.array([[1, 2, 3], [4, 5, 6]])

    theta_mean, phi_mean = _mean_coordinates(vertices)

    assert theta_mean == 2.5
    assert phi_mean == 3.5


def test_minmax_coordinates(mode_mono):
    vertices = np.array([[-1, -2, 3], [0, 7, 4], [6, 2, 2]])

    (theta_min, theta_max), (phi_min, phi_max) = _minmax_coordinates(vertices)

    assert theta_min == -1
    assert theta_max == 6
    assert phi_min == -2
    assert phi_max == 7


def test_vertex_index(mode_mono):
    assert _vertex_index(x=3, y=4, len_y=34) == 3 * 34 + 4


def test_transform_vertices_plane_parallel(mode_mono):
    vertices = np.array([[-1, -1, 2], [-1, 1, 2], [1, -1, 2], [1, 1, 2]])

    vertices_new = _transform_vertices_plane_parallel(vertices, 2, 10)

    expected_vertices = [
        [-0.0349, -0.0349, 12],
        [-0.0349, 0.0349, 12],
        [0.0349, -0.0349, 12],
        [0.0349, 0.0349, 12],
    ]

    for exp in expected_vertices:
        assert any([np.allclose(exp, v, atol=1e-4) for v in vertices_new])


def test_transform_vertices_spherical_shell(mode_mono):
    vertices = np.array([[-90, 0, 2], [90, 0, 2], [0, 0, 2], [0, 180, 2]])

    vertices_new = _transform_vertices_spherical_shell(vertices, 200)

    expected_vertices = [[0, 0, 202], [0, 0, -202], [202, 0, 0], [-202, 0, 0]]

    for exp in expected_vertices:
        assert any([np.allclose(exp, v, atol=1e-5) for v in vertices_new])


def test_to_uv(mode_mono):
    lat_min = -10
    lat_max = 10
    lon_min = 10
    lon_max = 30

    trafo = _to_uv(lat_min, lat_max, lon_min, lon_max)

    expected_trafo = mi.ScalarTransform4f.scale(
        (6, 3, 1)
    ) @ mi.ScalarTransform4f.translate(
        (-0.55555555556 + (0.5 / 6.0), -0.5 + (0.5 / 3.0), 0)
    )

    assert np.allclose(trafo.matrix, expected_trafo.matrix, rtol=1e-3, atol=1e-4)
