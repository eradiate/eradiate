import os
import tempfile

import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import FileMeshShape


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


def test_construct_file_ply(modes_all_double, tempfile_ply):

    ctx = KernelDictContext()
    mesh = FileMeshShape(filename=tempfile_ply)
    assert mesh.kernel_dict(ctx=ctx)
    kd = mesh.kernel_dict(ctx=ctx)
    assert kd[mesh.id]["type"] == "ply"
    assert kd.load(strip=False)


def test_construct_file_obj(modes_all_double, tempfile_obj):

    ctx = KernelDictContext()
    mesh = FileMeshShape(filename=tempfile_obj)
    assert mesh.kernel_dict(ctx=ctx)
    kd = mesh.kernel_dict(ctx=ctx)
    assert kd[mesh.id]["type"] == "obj"
    assert kd.load(strip=False)


def test_construct_file_illegal(modes_all_double):

    with pytest.raises(ValueError):
        ctx = KernelDictContext()
        mesh = FileMeshShape(filename="path/to/file.wrong")
