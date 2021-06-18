import os
import tempfile

import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere import MeshTree, MeshTreeElement
from eradiate.scenes.core import KernelDict

# -- Fixture definitions -------------------------------------------------------


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


@pytest.fixture(scope="module")
def tempfile_ply():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_mesh.ply")
        with open(filename, "w") as tf:
            tf.write(
                """ply 
format ascii 1.0 
element vertex 8 
property float x 
property float y 
property float z 
element face 12 
property list uchar int32 vertex_index
end_header 
0 0 0 
0 0 1 
0 1 1 
0 1 0 
1 0 0 
1 0 1 
1 1 1 
1 1 0  
3 0 1 2
3 0 2 3
3 7 6 5
3 7 5 4 
3 0 4 5
3 0 5 1 
3 1 5 6
3 1 6 2 
3 2 6 7
3 2 7 3 
3 3 7 4
3 3 4 0
"""
            )
        yield filename


@pytest.fixture(scope="module")
def tempfile_stl():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_mesh.stl")
        with open(filename, "w") as tf:
            tf.write("fake content\n")
        yield filename


def test_mesh_tree_element_instantiate(
    mode_mono, tempfile_obj, tempfile_ply, tempfile_stl
):

    # empty constructor raises due to missing mesh description file
    with pytest.raises(ValueError):
        MeshTreeElement()

    # obj file instantiation
    assert MeshTreeElement(
        id="mesh_tree_obj",
        mesh_filename=tempfile_obj,
        mesh_units=ureg.m,
        mesh_reflectance=0.5,
        mesh_transmittance=0.5,
    )

    # ply file instantiation
    assert MeshTreeElement(
        id="mesh_tree_ply",
        mesh_filename=tempfile_ply,
        mesh_units=ureg.m,
        mesh_reflectance=0.5,
        mesh_transmittance=0.5,
    )

    # unsupported file format raise
    with pytest.raises(ValueError):
        MeshTreeElement(
            id="mesh_tree_stl",
            mesh_filename=tempfile_stl,
            mesh_units=ureg.m,
            mesh_reflectance=0.5,
            mesh_transmittance=0.5,
        )


def test_mesh_tree_element_load(mode_mono, tempfile_obj, tempfile_ply):
    """
    Instantiate MeshTreeElement objects from obj and ply files and load the
    corresponding Mitsuba objects.
    """
    ctx = KernelDictContext()

    # obj file instantiation
    obj_tree = MeshTreeElement(
        id="mesh_tree_obj",
        mesh_filename=tempfile_obj,
        mesh_units=ureg.m,
        mesh_reflectance=0.5,
        mesh_transmittance=0.5,
    )

    assert KernelDict.new(obj_tree, ctx=ctx).load()

    # ply file instantiation
    ply_tree = MeshTreeElement(
        id="mesh_tree_ply",
        mesh_filename=tempfile_ply,
        mesh_units=ureg.m,
        mesh_reflectance=0.5,
        mesh_transmittance=0.5,
    )

    assert KernelDict.new(ply_tree, ctx=ctx).load()


def test_mesh_tree_instantiate(mode_mono, tempfile_obj):
    """
    Instantiate a MeshTree object holding two MeshTreeElements and load the
    corresponding Mitsuba objects.
    """
    ctx = KernelDictContext()

    # Constructor based instantiation
    assert MeshTree(
        mesh_tree_elements=[
            MeshTreeElement(
                id="mesh_tree_obj",
                mesh_filename=tempfile_obj,
                mesh_units=ureg.m,
                mesh_reflectance=0.5,
                mesh_transmittance=0.5,
            ),
            MeshTreeElement(
                id="mesh_tree_obj_2",
                mesh_filename=tempfile_obj,
                mesh_units=ureg.m,
                mesh_reflectance=0.1,
                mesh_transmittance=0.9,
            ),
        ]
    )

    tree = MeshTree(
        mesh_tree_elements=[
            MeshTreeElement(
                id="mesh_tree_obj",
                mesh_filename=tempfile_obj,
                mesh_units=ureg.m,
                mesh_reflectance=0.5,
                mesh_transmittance=0.5,
            ),
            MeshTreeElement(
                id="mesh_tree_obj_2",
                mesh_filename=tempfile_obj,
                mesh_units=ureg.m,
                mesh_reflectance=0.1,
                mesh_transmittance=0.9,
            ),
        ]
    )

    assert KernelDict.new(tree, ctx=ctx).load()

    # dict API based instantiation

    assert MeshTree.from_dict(
        {
            "id": "mesh_tree",
            "mesh_tree_elements": [
                {
                    "id": "mesh_tree_obj",
                    "mesh_filename": tempfile_obj,
                    "mesh_reflectance": 0.5,
                    "mesh_transmittance": 0.5,
                    "mesh_units": ureg.m,
                },
                {
                    "id": "mesh_tree_obj_2",
                    "mesh_filename": tempfile_obj,
                    "mesh_reflectance": 0.1,
                    "mesh_transmittance": 0.9,
                    "mesh_units": ureg.m,
                },
            ],
        }
    )

    tree = MeshTree.from_dict(
        {
            "id": "mesh_tree",
            "mesh_tree_elements": [
                {
                    "id": "mesh_tree_obj",
                    "mesh_filename": tempfile_obj,
                    "mesh_reflectance": 0.5,
                    "mesh_transmittance": 0.5,
                    "mesh_units": ureg.m,
                },
                {
                    "id": "mesh_tree_obj_2",
                    "mesh_filename": tempfile_obj,
                    "mesh_reflectance": 0.1,
                    "mesh_transmittance": 0.9,
                    "mesh_units": ureg.m,
                },
            ],
        }
    )

    assert KernelDict.new(tree, ctx=ctx).load()
