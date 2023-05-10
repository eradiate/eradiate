import os
import tempfile
from copy import deepcopy

import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

from eradiate import KernelContext
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.shapes import BufferMeshShape, FileMeshShape, Shape
from eradiate.scenes.surface import DEMSurface
from eradiate.units import unit_registry as ureg


@pytest.fixture(scope="module")
def tempfile_obj():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_mesh.obj")
        with open(filename, "w") as tf:
            tf.write("v 1 0 0\n")
            tf.write("v 0 1 0\n")
            tf.write("v 0 0 1\n")
            tf.write("f 1 2 3\n")
        yield filename


@pytest.mark.parametrize(
    "constructor, kwargs, expected_shape, expected_traversal_param_keys",
    [
        (
            None,
            {
                "shape": {"type": "file_mesh", "filename": "tempfile_obj"},
                "bsdf": {"type": "lambertian"},
            },
            FileMeshShape,
            {"terrain_shape.bsdf.reflectance.value"},
        ),
        (
            DEMSurface.from_dataarray,
            {
                "data": xr.DataArray(
                    data=np.zeros((10, 10)),
                    dims=["x", "y"],
                    coords={
                        "lat": (["x"], np.linspace(-5, 5.1, 10), {"units": "degree"}),
                        "lon": (["y"], np.linspace(-6, 6, 10), {"units": "degree"}),
                    },
                ),
                "bsdf": {"type": "lambertian"},
            },
            BufferMeshShape,
            {"terrain_shape.bsdf.reflectance.value"},
        ),
        (
            DEMSurface.from_analytical,
            {
                "elevation_function": lambda x, y: 10.0,
                "x_length": 10 * ureg.m,
                "y_length": 10 * ureg.m,
                "x_steps": 100,
                "y_steps": 100,
                "bsdf": {"type": "lambertian"},
            },
            BufferMeshShape,
            {"terrain_shape.bsdf.reflectance.value"},
        ),
    ],
    ids=["minimal", "from_dataarray", "from_analytical"],
)
def test_dem_surface_construct(
    modes_all_double,
    constructor,
    kwargs,
    expected_shape,
    expected_traversal_param_keys,
    request,
):
    # Expand fixtures
    kwargs = deepcopy(kwargs)

    try:
        if kwargs["shape"]["type"] == "file_mesh":
            kwargs["shape"]["filename"] = request.getfixturevalue(
                kwargs["shape"]["filename"]
            )
    except KeyError:
        pass

    # Instantiate DEM surface
    if constructor is None:
        surface = DEMSurface(**kwargs)
    else:
        surface = constructor(**kwargs)

    if issubclass(expected_shape, Shape):
        assert isinstance(surface.shape, expected_shape)
    else:
        raise NotImplementedError

    if isinstance(expected_traversal_param_keys, set):
        template, params = traverse(surface)

        # Scene element is composite: template has not "type" key
        assert "type" not in template
        # Parameter map keys are fetched recursively
        assert set(params.keys()) == expected_traversal_param_keys

        # When enclosed in a Scene, the surface can be traversed
        scene = Scene(objects={"surface": surface})
        template, params = traverse(scene)
        kernel_dict = template.render(KernelContext())
        assert isinstance(mi.load_dict(kernel_dict), mi.Scene)

    else:
        raise NotImplementedError
