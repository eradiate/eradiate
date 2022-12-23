import os
import tempfile

import mitsuba as mi
import pytest

from eradiate.scenes.shapes import FileMeshShape
from eradiate.test_tools.types import check_scene_element


@pytest.fixture(scope="module")
def tempfile_ply():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_mesh.ply")
        with open(filename, "w") as tf:
            tf.write("ply\n")
            tf.write("format ascii 1.0\n")
            tf.write("element vertex 3\n")
            tf.write("property float x\n")
            tf.write("property float y\n")
            tf.write("property float z\n")
            tf.write("element face 1\n")
            tf.write("property list uchar int vertex_index\n")
            tf.write("end_header\n")
            tf.write("1 0 0\n")
            tf.write("0 1 0\n")
            tf.write("0 0 1\n")
            tf.write("3 1 2 3\n")
        yield filename


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
    "file_type, tempfile",
    [
        ("ply", "tempfile_ply"),
        ("obj", "tempfile_obj"),
    ],
    ids=["ply", "obj"],
)
def test_filemesh_kernel_dict(modes_all_double, file_type, tempfile, request):
    mesh = FileMeshShape(filename=request.getfixturevalue(tempfile))
    check_scene_element(mesh, mi.Mesh)


def test_construct_file_illegal(modes_all_double):
    with pytest.raises(ValueError, match="mesh files only in PLY or OBJ format"):
        FileMeshShape(filename="path/to/file.wrong")
