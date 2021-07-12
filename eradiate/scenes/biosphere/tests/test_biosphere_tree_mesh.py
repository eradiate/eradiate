import os
import tempfile

import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere import MeshTree, MeshTreeElement
from eradiate.scenes.core import KernelDict

# ------------------------------------------------------------------------------
#                            Fixture definitions
# ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmpfile_obj():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tmpfile_mesh.obj")
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
def tmpfile_ply():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tmpfile_mesh.ply")
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
def tmpfile_stl():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tmpfile_mesh.stl")
        with open(filename, "w") as tf:
            tf.write("fake content\n")
        yield filename


# ------------------------------------------------------------------------------
#                                   Test
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tmp_file",
    ["tmpfile_obj", "tmpfile_ply", "tmpfile_stl"],
    ids=["obj", "ply", "stl"],
)
def test_mesh_tree_element_instantiate(mode_mono, tmp_file, request):
    tmp_file = request.getfixturevalue(tmp_file)

    # empty constructor raises due to missing mesh description file
    with pytest.raises(ValueError):
        MeshTreeElement()

    # instantiation with supported format succeeds, except for stl
    if "tmpfile_stl" not in request.fixturenames:
        assert MeshTreeElement(
            id="mesh_tree",
            mesh_filename=tmp_file,
            mesh_units=ureg.m,
            reflectance=0.5,
            transmittance=0.5,
        )
    else:
        with pytest.raises(ValueError):
            MeshTreeElement(
                id="mesh_tree_stl",
                mesh_filename=tmp_file,
                mesh_units=ureg.m,
                reflectance=0.5,
                transmittance=0.5,
            )


@pytest.mark.parametrize("tmp_file", ["tmpfile_obj", "tmpfile_ply"], ids=["obj", "ply"])
def test_mesh_tree_element_load(mode_mono, tmp_file, request):
    """
    Instantiate MeshTreeElement objects from obj and ply files and load the
    corresponding Mitsuba objects.
    """
    tmp_file = request.getfixturevalue(tmp_file)
    ctx = KernelDictContext(ref=True)

    tree_element = MeshTreeElement(
        id="mesh_tree_obj",
        mesh_filename=tmp_file,
        mesh_units=ureg.m,
        reflectance=0.5,
        transmittance=0.5,
    )
    d = {**tree_element.bsdfs(ctx), **tree_element.shapes(ctx)}

    assert KernelDict.new(d, ctx=ctx).load()


def test_mesh_tree_instantiate(mode_mono, tmpfile_obj):
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
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.5,
                transmittance=0.5,
            ),
            MeshTreeElement(
                id="mesh_tree_obj_2",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.1,
                transmittance=0.9,
            ),
        ]
    )

    tree = MeshTree(
        mesh_tree_elements=[
            MeshTreeElement(
                id="mesh_tree_obj",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.5,
                transmittance=0.5,
            ),
            MeshTreeElement(
                id="mesh_tree_obj_2",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.1,
                transmittance=0.9,
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
                    "mesh_filename": tmpfile_obj,
                    "reflectance": 0.5,
                    "transmittance": 0.5,
                    "mesh_units": ureg.m,
                },
                {
                    "id": "mesh_tree_obj_2",
                    "mesh_filename": tmpfile_obj,
                    "reflectance": 0.1,
                    "transmittance": 0.9,
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
                    "mesh_filename": tmpfile_obj,
                    "reflectance": 0.5,
                    "transmittance": 0.5,
                    "mesh_units": ureg.m,
                },
                {
                    "id": "mesh_tree_obj_2",
                    "mesh_filename": tmpfile_obj,
                    "reflectance": 0.1,
                    "transmittance": 0.9,
                    "mesh_units": ureg.m,
                },
            ],
        }
    )

    assert KernelDict.new(tree, ctx=ctx).load()
