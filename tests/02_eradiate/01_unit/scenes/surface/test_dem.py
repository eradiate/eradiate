import os
import tempfile

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest
import xarray as xr

from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.shapes import FileMeshShape
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


def test_dem_construct_default(modes_all_double, tempfile_obj):

    ctx = KernelDictContext()

    dem = DEMSurface(
        id="dem", shape=FileMeshShape(filename=tempfile_obj, bsdf=LambertianBSDF())
    )

    assert dem.kernel_dict(ctx=ctx).load(strip=False)


def test_dem_construct_dataarray(modes_all_double, tempfile_obj):

    ctx = KernelDictContext()

    da = xr.DataArray(
        data=np.zeros((10, 10)),
        dims=["x", "y"],
        coords=dict(
            lat=(["x"], np.linspace(-5, 5.1, 10)),
            lon=(["y"], np.linspace(-6, 6, 10)),
        ),
    )
    da.lat.attrs["units"] = "degree"
    da.lon.attrs["units"] = "degree"

    dem = DEMSurface.from_dataarray(data=da, bsdf=LambertianBSDF())
    assert dem.kernel_dict(ctx=ctx).load(strip=False)


def test_dem_construct_analytical(modes_all_double, tempfile_obj):

    # Default constructor
    ctx = KernelDictContext()

    def elevation(x, y):
        return 10.0

    dem = DEMSurface.from_analytical(
        elevation,
        x_length=10 * ureg.m,
        y_length=10 * ureg.m,
        x_steps=100,
        y_steps=100,
        bsdf=LambertianBSDF(),
    )
    assert dem.kernel_dict(ctx=ctx).load(strip=False)
