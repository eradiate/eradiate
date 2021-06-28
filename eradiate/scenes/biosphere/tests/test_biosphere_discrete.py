import os
import tempfile

import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere._discrete import (
    DiscreteCanopy,
    InstancedCanopyElement,
    LeafCloud,
)
from eradiate.scenes.biosphere._tree import AbstractTree
from eradiate.scenes.core import KernelDict

# -- Fixture definitions -------------------------------------------------------


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


# -- InstancedCanopyElement tests ----------------------------------------------


def test_instanced_canopy_element_create(mode_mono):
    """Unit testing for the :class:`InstancedCanopyElement` constructor."""
    cloud = LeafCloud(
        leaf_positions=[[0, 0, 0]], leaf_orientations=[[0, 0, 1]], leaf_radii=[0.1]
    )
    tree = AbstractTree(
        leaf_cloud=cloud, trunk_height=1.0, trunk_radius=0.1, trunk_reflectance=0.5
    )
    positions = np.array([[-5, -5, -5], [5, 5, 5]])
    assert InstancedCanopyElement(canopy_element=cloud, instance_positions=positions)
    assert InstancedCanopyElement(canopy_element=tree, instance_positions=positions)


def test_instanced_canopy_element_kernel_dict(mode_mono):
    """Unit testing for :meth:`InstancedLeafCloud.kernel_dict`."""
    ctx = KernelDictContext()

    cloud = LeafCloud(
        leaf_positions=[[0, 0, 0]],
        leaf_orientations=[[0, 0, 1]],
        leaf_radii=[0.1],
        id="leaf_cloud",
    )
    tree = AbstractTree(
        leaf_cloud=cloud,
        trunk_height=1.0,
        trunk_radius=0.1,
        trunk_reflectance=0.5,
        id="tree",
    )
    positions = np.array([[-5, -5, -5], [5, 5, 5]])
    kernel_dict = InstancedCanopyElement(
        canopy_element=tree, instance_positions=positions, id="element"
    ).kernel_dict(ctx=ctx)

    # The generated kernel dictionary can be instantiated
    assert KernelDict.new(kernel_dict).load()


def test_instanced_leaf_cloud_from_file(mode_mono, tempfile_spheres):
    """Unit testing for :meth:`InstancedLeafCloud.from_file`."""
    assert InstancedCanopyElement.from_file(
        filename=tempfile_spheres,
        canopy_element=LeafCloud.from_dict(
            {
                "leaf_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            }
        ),
    )


def test_instanced_canopy_element_from_dict(mode_mono, tempfile_spheres):
    """Unit testing for :meth:`InstancedLeafCloud.from_dict`."""
    ctx = KernelDictContext()

    # We can instantiate from a full-dict spec
    instanced_leaf_cloud = InstancedCanopyElement.from_dict(
        {
            "canopy_element": {
                "type": "leaf_cloud",
                "leaf_positions": [[0, 0, 0], [1, 1, 1]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            },
            "instance_positions": np.array([[-5, -5, -5], [5, 5, 5]]),
        }
    )
    assert instanced_leaf_cloud

    # The generated kernel dictionary can be instantiated
    assert KernelDict.new(instanced_leaf_cloud.kernel_dict(ctx=ctx)).load()

    # We can access the from_file constructor from a dict
    assert InstancedCanopyElement.from_dict(
        {
            "construct": "from_file",
            "filename": tempfile_spheres,
            "canopy_element": {
                "type": "leaf_cloud",
                "leaf_positions": [[0, 0, 0], [1, 1, 1]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            },
        }
    )


# -- DiscreteCanopy tests ------------------------------------------------------


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

    # We can reproduce this behaviour using the dict-based API
    canopy = DiscreteCanopy.from_dict(
        {
            "construct": "homogeneous",
            "n_leaves": 1,
            "leaf_radius": 0.1,
            "l_horizontal": 10,
            "l_vertical": 3,
        }
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

    # We can reproduce this behaviour using the dict-based API
    canopy = DiscreteCanopy.from_dict(
        {
            "construct": "leaf_cloud_from_files",
            "size": [1, 1, 1],
            "leaf_cloud_dicts": [
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
        }
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_advanced(mode_mono, tempfile_spheres, tempfile_leaves):
    """
    A more advanced test where we load a pre-computed canopy consisting of a
    generated cuboid leaf cloud and a series of instanced pre-computed leaf
    clouds.
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
        ],
    )
    assert KernelDict.new(canopy, ctx=ctx).load()

    # Reproduce previous example with dict API
    canopy = DiscreteCanopy.from_dict(
        {
            "size": [1.0, 1.0, 1.0] * ureg.m,
            "instanced_canopy_elements": [
                {
                    "instance_positions": [[0, 0, 0]],
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "cuboid",
                        "n_leaves": 1,
                        "leaf_radius": 0.1,
                        "l_horizontal": 10,
                        "l_vertical": 3,
                        "id": "leaf_cloud_cuboid",
                    },
                },
                {
                    "construct": "from_file",
                    "filename": tempfile_spheres,
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "from_file",
                        "filename": tempfile_leaves,
                        "id": "leaf_cloud_precomputed",
                    },
                },
            ],
        }
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
    padded_canopy = canopy.padded(2)
    assert padded_canopy
    assert KernelDict.new(padded_canopy, ctx=ctx).load()
    # Padded canopy has (2*padding + 1) ** 2 times more instances than original
    assert len(padded_canopy.instances()) == len(canopy.instances()) * 25
    # Padded canopy has 2*padding + 1 times larger horizontal size than original
    assert np.allclose(padded_canopy.size[:2], 5 * canopy.size[:2])
    # Padded canopy has same vertical size as original
    assert padded_canopy.size[2] == canopy.size[2]
