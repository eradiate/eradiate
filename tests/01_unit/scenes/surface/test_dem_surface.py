import mitsuba as mi
import numpy as np
import pint
import pytest
import xarray as xr

from eradiate import KernelContext
from eradiate.constants import EARTH_RADIUS
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.geometry import SphericalShellGeometry
from eradiate.scenes.shapes import RectangleShape, SphereShape
from eradiate.scenes.surface import DEMSurface
from eradiate.scenes.surface._dem import (
    _transform_vertices_spherical_shell_lonlat,
    mesh_from_dem,
    triangulate_grid,
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
                        "lon": np.linspace(-0.1, 0.1, 10) * ureg.deg,
                        "lat": np.linspace(-0.1, 0.1, 10) * ureg.deg,
                    },
                ),
                "geometry": "plane_parallel",
                "planet_radius": EARTH_RADIUS,
            },
            [
                (-11131.8845, 11131.8845) * ureg.m,
                (-11131.8845, 11131.8845) * ureg.m,
            ],
        ),
        (
            {
                "da": make_dataarray(
                    data=np.zeros((10, 10)) * ureg.m,
                    coords={
                        "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
                        "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
                    },
                ),
                "geometry": "spherical_shell",
                "planet_radius": EARTH_RADIUS,
            },
            [(-0.4, 0.5) * ureg.deg, (-0.2, 0.3) * ureg.deg],
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
            [(-20.0, 20.0) * ureg.km, (-30.0, 30.0) * ureg.km],
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
            [(-0.089832, 0.179664) * ureg.deg, (-0.269496, 0.359328) * ureg.deg],
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
                "geometry": "invalidgeometry",
                "planet_radius": EARTH_RADIUS,
            },
            None,
        ),
    ],
    ids=[
        "plane_parallel_lonlat",
        "spherical_shell_lonlat",
        "plane_parallel_xy",
        "spherical_shell_xy",
        "invalid_geometry",
    ],
)
def test_mesh_from_dem(mode_mono, kwargs, expected_limits):
    if kwargs["geometry"] not in ["spherical_shell", "plane_parallel"]:
        with pytest.raises(ValueError):
            mesh_from_dem(**kwargs)
    else:
        if (kwargs["geometry"] == "spherical_shell") and kwargs[
            "planet_radius"
        ] is not None:
            with pytest.warns():
                mesh, xlon_lim, ylat_lim = mesh_from_dem(**kwargs)
        else:
            mesh, xlon_lim, ylat_lim = mesh_from_dem(**kwargs)

        assert len(mesh.vertices) == 100

        assert np.allclose(xlon_lim, expected_limits[0])
        assert np.allclose(ylat_lim, expected_limits[1])


@pytest.mark.parametrize(
    "divide, expected_faces",
    [
        (
            "nesw",
            [
                [0, 1, 5],
                [1, 2, 6],
                [2, 3, 7],
                [4, 5, 9],
                [5, 6, 10],
                [6, 7, 11],
                [0, 5, 4],
                [1, 6, 5],
                [2, 7, 6],
                [4, 9, 8],
                [5, 10, 9],
                [6, 11, 10],
            ],
        ),
        (
            "nwse",
            [
                [0, 4, 1],
                [1, 5, 2],
                [2, 6, 3],
                [4, 8, 5],
                [5, 9, 6],
                [6, 10, 7],
                [4, 5, 1],
                [5, 6, 2],
                [6, 7, 3],
                [8, 9, 5],
                [9, 10, 6],
                [10, 11, 7],
            ],
        ),
    ],
    ids=["nesw", "nwse"],
)
def test_triangulate_grid(divide, expected_faces):
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 3)

    expected_vertices = [
        [-1, -1],
        [-1 / 3, -1],
        [1 / 3, -1],
        [1, -1],
        [-1, 0],
        [-1 / 3, 0],
        [1 / 3, 0],
        [1, 0],
        [-1, 1],
        [-1 / 3, 1],
        [1 / 3, 1],
        [1, 1],
    ]

    vertices, faces = triangulate_grid(x, y, divide=divide)
    np.testing.assert_almost_equal(vertices, expected_vertices)
    np.testing.assert_equal(faces, expected_faces)


def test_transform_vertices_spherical_shell(mode_mono):
    vertices = np.array(
        [
            [-0.5 * np.pi, -0.25 * np.pi, 1],
            [0.5 * np.pi, 0.25 * np.pi, 1],
            [-np.pi, -0.5 * np.pi, 1],
            [-np.pi, 0.5 * np.pi, 1],
            [np.pi, -0.5 * np.pi, 1],
            [np.pi, 0.5 * np.pi, 1],
        ]
    )
    vertices = _transform_vertices_spherical_shell_lonlat(vertices, 10)

    expected = [
        [-7.778175, -7.778175, 0],
        [7.778175, 7.778175, 0],
        [0, -11, 0],
        [0, 11, 0],
        [0, -11, 0],
        [0, 11, 0],
    ]

    np.testing.assert_allclose(vertices, expected, atol=1e-8)


@pytest.mark.parametrize(
    "geometry, expected_bg_cls, planet_radius",
    [
        ("plane_parallel", RectangleShape, EARTH_RADIUS),
        ("spherical_shell", SphereShape, None),
    ],
)
def test_dem_surface_construct_from_mesh(
    modes_all_mono, geometry, expected_bg_cls, planet_radius
):
    mesh, xlon_lim, ylat_lim = mesh_from_dem(
        da=make_dataarray(
            data=np.zeros((10, 10)) * ureg.m,
            coords={
                "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
                "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
            },
        ),
        geometry=geometry,
        planet_radius=planet_radius,
    )

    dem = DEMSurface.from_mesh(
        id="terrain", mesh=mesh, xlon_lim=xlon_lim, ylat_lim=ylat_lim, geometry=geometry
    )
    assert isinstance(dem.shape_background, expected_bg_cls)


def test_dem_surface_kernel_dict(mode_mono):
    geometry = SphericalShellGeometry()
    mesh, xlon_lim, ylat_lim = mesh_from_dem(
        da=make_dataarray(
            data=np.zeros((10, 10)) * ureg.m,
            coords={
                "lon": np.linspace(-0.4, 0.5, 10) * ureg.deg,
                "lat": np.linspace(-0.2, 0.3, 10) * ureg.deg,
            },
        ),
        geometry=geometry,
    )

    dem = DEMSurface.from_mesh(
        id="terrain", mesh=mesh, xlon_lim=xlon_lim, ylat_lim=ylat_lim, geometry=geometry
    )

    check_scene_element(dem)

    # When enclosed in a Scene, the surface can be traversed
    scene = Scene(objects={"surface": dem})
    template, params = traverse(scene)
    kernel_dict = template.render(KernelContext())
    assert isinstance(mi.load_dict(kernel_dict), mi.Scene)
