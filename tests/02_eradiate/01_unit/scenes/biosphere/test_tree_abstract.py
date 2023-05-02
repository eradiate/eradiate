import os
import tempfile

import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.scenes.biosphere import AbstractTree, LeafCloud
from eradiate.scenes.core import traverse
from eradiate.test_tools.types import check_scene_element

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


# ------------------------------------------------------------------------------
#                              AbstractTree Tests
# ------------------------------------------------------------------------------


def test_abstract_tree_instantiate(mode_mono):
    """Unit tests for :class:`AbstractTree`'s default constructor."""

    # Empty constructor does not raise
    assert AbstractTree()

    # Now with more sensible values
    assert AbstractTree(
        leaf_cloud=LeafCloud(
            leaf_positions=[[0, 0, 0]], leaf_orientations=[[0, 0, 1]], leaf_radii=[0.1]
        ),
        trunk_height=0.5,
        trunk_radius=0.1,
        trunk_reflectance=0.5,
    )


def test_abstract_tree_dispatch_leaf_cloud(mode_mono, tempfile_leaves):
    """Test if contained LeafCloud is instantiated in all variants"""

    # A LeafCloud instance can be loaded from a file on the hard drive
    tree = AbstractTree(leaf_cloud=LeafCloud.from_file(tempfile_leaves))
    assert len(tree.leaf_cloud.leaf_positions) == 5
    assert np.allclose(tree.leaf_cloud.leaf_radii, 0.1 * ureg.m)
    # Produced kernel dict is valid
    check_scene_element(tree)
    # Check parameter map contents
    _, umap_template = traverse(tree)
    assert set(umap_template.keys()) == {
        "bsdf_leaf_cloud.reflectance.value",
        "bsdf_leaf_cloud.transmittance.value",
        "bsdf_abstract_tree.reflectance.value",
    }

    # When passing a dict for the leaf_cloud field, the 'type' param can be omitted
    assert AbstractTree(
        leaf_cloud={
            "leaf_positions": [[0, 0, 0], [1, 1, 1]],
            "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
            "leaf_radii": [0.1, 0.1],
        }
    )

    # Dispatch to from_file if requested
    tree1 = AbstractTree(leaf_cloud=LeafCloud.from_file(tempfile_leaves))
    tree2 = AbstractTree(
        leaf_cloud={"construct": "from_file", "filename": tempfile_leaves}
    )
    # assert leaf clouds are equal
    assert np.all(tree1.leaf_cloud.leaf_positions == tree2.leaf_cloud.leaf_positions)
    assert np.all(
        tree1.leaf_cloud.leaf_orientations == tree2.leaf_cloud.leaf_orientations
    )
    assert np.all(tree1.leaf_cloud.leaf_radii == tree2.leaf_cloud.leaf_radii)
    assert (
        tree1.leaf_cloud.leaf_transmittance.value
        == tree2.leaf_cloud.leaf_transmittance.value
    )
    assert (
        tree1.leaf_cloud.leaf_reflectance.value
        == tree2.leaf_cloud.leaf_reflectance.value
    )

    # Dispatch to generator if requested
    tree = AbstractTree(
        leaf_cloud={
            "construct": "cuboid",
            "n_leaves": 100,
            "l_horizontal": 10.0,
            "l_vertical": 1.0,
            "leaf_radius": 10.0,
            "leaf_radius_units": "cm",
        }
    )
    assert len(tree.leaf_cloud.leaf_radii) == 100
    assert len(tree.leaf_cloud.leaf_positions) == 100
    assert np.allclose(tree.leaf_cloud.leaf_radii, 10.0 * ureg.cm)


def test_abstract_tree_kernel_dict(mode_mono):
    """Partial unit testing for :meth:`LeafCloud.kernel_dict`."""
    tree_id = "my_tree"
    cloud_id = "my_cloud"
    tree = AbstractTree(
        leaf_cloud=LeafCloud(
            id=cloud_id,
            leaf_positions=[[0, 0, 0], [1, 1, 1]],
            leaf_orientations=[[1, 0, 0], [0, 1, 0]],
            leaf_radii=[0.1, 0.1],
            leaf_reflectance=0.5,
            leaf_transmittance=0.5,
        ),
        id=tree_id,
        trunk_height=2.0,
        trunk_radius=0.2,
        trunk_reflectance=0.5,
    )

    template = traverse(tree)[0]
    kdict = template.render(ctx=KernelContext())

    # The BSDF is bilambertian with the parameters we initially set
    assert kdict[f"bsdf_{cloud_id}"] == {
        "type": "bilambertian",
        "reflectance": {"type": "uniform", "value": 0.5},
        "transmittance": {"type": "uniform", "value": 0.5},
    }

    # Leaves are disks
    for shape_key in [f"{cloud_id}_leaf_0", f"{cloud_id}_leaf_1"]:
        assert kdict[shape_key]["type"] == "disk"

    # Kernel dict is valid
    check_scene_element(tree)
