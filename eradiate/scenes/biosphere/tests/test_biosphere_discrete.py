import os
import tempfile

import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere import (
    AbstractTree,
    InstancedCanopyElement,
    MeshTree,
    MeshTreeElement,
)
from eradiate.scenes.biosphere._discrete import DiscreteCanopy, LeafCloud
from eradiate.scenes.core import KernelDict

# ------------------------------------------------------------------------------
#                            Fixture definitions
# ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tempfile_leaves():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_leaves.txt")
        with open(filename, "w") as tf:
            tf.write("0.100 8.864 9.040 1.878 -0.314 0.025 0.949\n")
            tf.write("0.100 9.539 -10.463 0.627 0.489 -0.276 0.828\n")
            tf.write("0.100 -2.274 -9.204 0.797 0.618 0.184 0.764\n")
            tf.write("0.100 -9.957 -4.971 0.719 -0.066 0.100 0.993\n")
            tf.write("0.100 5.339 9.153 0.500 0.073 -0.294 0.953\n")
        yield filename


@pytest.fixture(scope="module")
def tempfile_spheres():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_leaves.txt")
        with open(filename, "w") as tf:
            tf.write("8.864 9.040 1.878\n")
            tf.write("0.100 9.539 -10.463\n")
            tf.write("0.100 -2.274 -9.204\n")
            tf.write("0.100 -9.957 -4.971\n")
            tf.write("0.100 5.339 9.153\n")
        yield filename


@pytest.fixture(scope="module")
def tempfile_obj():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_mesh.obj")
        with open(filename, "w") as tf:
            tf.write(
                """o Cube
v 1.000000 -1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v -1.000000 -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v 1.000000 1.000000 -0.999999
v 0.999999 1.000000 1.000001
v -1.000000 1.000000 1.000000
v -1.000000 1.000000 -1.000000
vt 1.000000 0.333333
vt 1.000000 0.666667
vt 0.666667 0.666667
vt 0.666667 0.333333
vt 0.666667 0.000000
vt 0.000000 0.333333
vt 0.000000 0.000000
vt 0.333333 0.000000
vt 0.333333 1.000000
vt 0.000000 1.000000
vt 0.000000 0.666667
vt 0.333333 0.333333
vt 0.333333 0.666667
vt 1.000000 0.000000
vn 0.000000 -1.000000 0.000000
vn 0.000000 1.000000 0.000000
vn 1.000000 0.000000 0.000000
vn -0.000000 0.000000 1.000000
vn -1.000000 -0.000000 -0.000000
vn 0.000000 0.000000 -1.000000
s off
f 2/1/1 3/2/1 4/3/1
f 8/1/2 7/4/2 6/5/2
f 5/6/3 6/7/3 2/8/3
f 6/8/4 7/5/4 3/4/4
f 3/9/5 7/10/5 8/11/5
f 1/12/6 4/13/6 8/11/6
f 1/4/1 2/1/1 4/3/1
f 5/14/2 8/1/2 6/5/2
f 1/12/3 5/6/3 2/8/3
f 2/12/4 6/8/4 3/4/4
f 4/13/5 3/9/5 8/11/5
f 5/6/6 1/12/6 8/11/6"""
            )
        yield filename


# ------------------------------------------------------------------------------
#                             DiscreteCanopy Tests
# ------------------------------------------------------------------------------


def test_discrete_canopy_instantiate(mode_mono):
    # Constructing an empty canopy is possible
    assert DiscreteCanopy()


def test_discrete_canopy_homogeneous(mode_mono):
    ctx = KernelDictContext()

    # The generate_homogeneous() constructor returns a valid canopy object
    canopy = DiscreteCanopy.homogeneous(
        n_leaves=1, leaf_radius=0.1, l_horizontal=10, l_vertical=3
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_from_files(mode_mono, tempfile_spheres, tempfile_leaves):
    ctx = KernelDictContext()

    # The from_files() constructor returns a valid canopy object
    canopy = DiscreteCanopy.leaf_cloud_from_files(
        size=[1, 1, 1],
        leaf_cloud_dicts=[
            {
                "sub_id": "spheres_1",
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
            },
            {
                "sub_id": "spheres_2",
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
            },
        ],
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_advanced(
    mode_mono, tempfile_spheres, tempfile_leaves, tempfile_obj
):
    """
    A more advanced test where we load a series of different canopy elements:

    - A pre-computed canopy consisting of a generated cuboid leaf cloud
    - A series of instanced leaf clouds from files
    - An abstract tree with a leaf cloud
    - A mesh based canopy element

    """

    ctx = KernelDictContext()

    # First use the regular Python API
    canopy = DiscreteCanopy(
        size=[1.0, 1.0, 1.0] * ureg.m,
        instanced_canopy_elements=[
            InstancedCanopyElement(
                instance_positions=[[0, 0, 0]],
                canopy_element=LeafCloud.cuboid(
                    n_leaves=1,
                    leaf_radius=0.1,
                    l_horizontal=10,
                    l_vertical=3,
                    id="leaf_cloud_cuboid",
                ),
            ),
            InstancedCanopyElement.from_file(
                filename=tempfile_spheres,
                canopy_element=LeafCloud.from_file(
                    filename=tempfile_leaves, id="leaf_cloud_precomputed"
                ),
            ),
            InstancedCanopyElement(
                instance_positions=[[0, 0, 0]],
                canopy_element=AbstractTree(
                    leaf_cloud=LeafCloud.cuboid(
                        n_leaves=100,
                        l_horizontal=10.0,
                        l_vertical=1.0,
                        leaf_radius=10.0 * ureg.cm,
                    ),
                    trunk_height=2.0,
                    trunk_radius=0.2,
                    trunk_reflectance=0.5,
                    id="abstract_tree",
                ),
            ),
            InstancedCanopyElement(
                instance_positions=[[0, 0, 0]],
                canopy_element=MeshTree(
                    id="mesh_tree",
                    mesh_tree_elements=[
                        MeshTreeElement(
                            mesh_filename=tempfile_obj,
                            mesh_units=ureg.m,
                            reflectance=0.5,
                            transmittance=0.5,
                            id="mesh_element",
                        )
                    ],
                ),
            ),
        ],
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_padded(mode_mono, tempfile_leaves, tempfile_spheres):
    """Unit tests for :meth:`.DiscreteCanopy.padded`"""
    ctx = KernelDictContext()

    canopy = DiscreteCanopy.leaf_cloud_from_files(
        id="canopy",
        size=[100, 100, 30],
        leaf_cloud_dicts=[
            {
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
                "sub_id": "leaf_cloud",
            }
        ],
    )
    # The padded canopy object is valid and instantiable
    padded_canopy = canopy.padded_copy(2)
    assert padded_canopy
    assert KernelDict.new(padded_canopy, ctx=ctx).load()
    # Padded canopy has (2*padding + 1) ** 2 times more instances than original
    assert len(padded_canopy.instances(ctx)) == len(canopy.instances(ctx)) * 25
    # Padded canopy has 2*padding + 1 times larger horizontal size than original
    assert np.allclose(padded_canopy.size[:2], 5 * canopy.size[:2])
    # Padded canopy has same vertical size as original
    assert padded_canopy.size[2] == canopy.size[2]
