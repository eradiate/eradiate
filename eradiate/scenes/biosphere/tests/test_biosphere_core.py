import os
import tempfile

import numpy as np
import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere import AbstractTree, InstancedCanopyElement, LeafCloud
from eradiate.scenes.core import KernelDict

# ------------------------------------------------------------------------------
#                            Fixture definitions
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
#                        InstancedCanopyElement Tests
# ------------------------------------------------------------------------------


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
    assert KernelDict(kernel_dict).load()


def test_instanced_leaf_cloud_from_file(mode_mono, tempfile_spheres):
    """Unit testing for :meth:`InstancedLeafCloud.from_file`."""
    assert InstancedCanopyElement.from_file(
        filename=tempfile_spheres,
        canopy_element={
            "type": "leaf_cloud",
            "leaf_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
            "leaf_radii": [0.1, 0.1],
        },
    )
