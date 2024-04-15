import os
import tempfile

import mitsuba as mi
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.kernel._kernel_dict import KernelDictTemplate
from eradiate.scenes.biosphere import MeshTree, MeshTreeElement
from eradiate.scenes.core import traverse
from eradiate.test_tools.types import check_scene_element
from eradiate.util.misc import flatten

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
#                                   Tests
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tmp_file",
    ["tmpfile_obj", "tmpfile_ply", "tmpfile_stl"],
    ids=["obj", "ply", "stl"],
)
def test_mesh_tree_element_instantiate(mode_mono, tmp_file, request):
    tmp_file = request.getfixturevalue(tmp_file)

    # empty constructor raises due to missing mesh description file
    with pytest.raises(TypeError):
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

    tree_element = MeshTreeElement(
        id="mesh_tree_obj",
        mesh_filename=tmp_file,
        mesh_units=ureg.m,
        reflectance=0.5,
        transmittance=0.5,
    )
    template = KernelDictTemplate(
        flatten(
            {
                "type": "scene",
                **tree_element._template_bsdfs,
                **tree_element._template_shapes,
            }
        )
    )

    # Check that the template successfully expands to a valid Mitsuba scene
    mi_scene = mi.load_dict(template.render(KernelContext()))
    assert isinstance(mi_scene, mi.Scene)


def test_mesh_tree_instantiate(mode_mono, tmpfile_obj):
    """
    Instantiate a MeshTree object holding two MeshTreeElements and load the
    corresponding Mitsuba objects.
    """

    # Constructor based instantiation
    assert MeshTree(
        mesh_tree_elements=[
            MeshTreeElement(
                id="foliage",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.5,
                transmittance=0.5,
            ),
            MeshTreeElement(
                id="trunk",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.9,
                transmittance=0.1,
            ),
        ]
    )

    tree = MeshTree(
        mesh_tree_elements=[
            MeshTreeElement(
                id="foliage",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.5,
                transmittance=0.5,
            ),
            MeshTreeElement(
                id="trunk",
                mesh_filename=tmpfile_obj,
                mesh_units=ureg.m,
                reflectance=0.9,
                transmittance=0.1,
            ),
        ]
    )

    check_scene_element(tree)
    _, params = traverse(tree)
    assert set(params.keys()) == {
        "bsdf_foliage.reflectance.value",
        "bsdf_foliage.transmittance.value",
        "bsdf_trunk.reflectance.value",
        "bsdf_trunk.transmittance.value",
    }

    # check that special converter for the mesh_tree_elements field works as
    # expected
    tree = MeshTree(
        id="mesh_tree",
        mesh_tree_elements=[
            {
                "id": "tree_foliage",
                "mesh_filename": tmpfile_obj,
                "mesh_units": "m",
                "reflectance": 0.5,
                "transmittance": 0.5,
            },
            {
                "id": "tree_trunk",
                "mesh_filename": tmpfile_obj,
                "mesh_units": "m",
                "reflectance": 0.9,
                "transmittance": 0.1,
            },
        ],
    )

    check_scene_element(tree)
    _, params = traverse(tree)
    assert set(params.keys()) == {
        "bsdf_tree_foliage.reflectance.value",
        "bsdf_tree_foliage.transmittance.value",
        "bsdf_tree_trunk.reflectance.value",
        "bsdf_tree_trunk.transmittance.value",
    }
