import mitsuba as mi
import numpy as np
import pytest

from eradiate.exceptions import TraversalError
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.shapes import BufferMeshShape
from eradiate.test_tools.types import check_scene_element


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"vertices": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "faces": [[1, 2, 3]]}, None),
        ({"vertices": [1, 0, 0, 0, 1, 0, 0, 0, 1], "faces": [[1, 2, 3]]}, ValueError),
        (
            {"vertices": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "faces": [1, 2, 3]},
            ValueError,
        ),
    ],
    ids=["correct", "flat_vertex", "flat_face"],
)
def test_buffer_mesh_construct(modes_all_double, kwargs, expected):
    if expected is None:
        assert BufferMeshShape(**kwargs)

    elif issubclass(expected, Exception):
        with pytest.raises(expected):
            BufferMeshShape(**kwargs)


def test_buffermesh_construct_trafo_warning(mode_mono):
    with pytest.raises(Warning):
        BufferMeshShape(
            vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[1, 2, 3]],
            to_world=mi.ScalarTransform4f().scale(2),
        )


def test_buffer_mesh_instance(mode_mono):
    mesh = BufferMeshShape(
        vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[1, 2, 3]],
        bsdf={"type": "lambertian"},
    )
    assert isinstance(mesh.instance, mi.Mesh)


def test_buffer_mesh_params(mode_mono):
    mesh = BufferMeshShape(
        vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[1, 2, 3]],
        bsdf={"type": "lambertian"},
    )
    assert set(mesh.params.keys()) == {"bsdf.reflectance.value"}


def test_buffer_mesh_traverse(mode_mono):
    mesh = BufferMeshShape(
        vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[1, 2, 3]],
        bsdf={"type": "lambertian"},
    )

    # Traversal should fail because instance nodes can't be traversed on their own
    with pytest.raises(
        TraversalError, match="Instances may only be inserted as child nodes"
    ):
        traverse(mesh)

    # Traversal as part of an enclosing object should succeed
    scene = Scene(objects={"mesh": mesh})
    check_scene_element(scene, mi.Scene)


@pytest.mark.parametrize(
    "texcoords", [None, [[0, 0], [0, 1], [1, 1], [1, 0]]], ids=["no_uvs", "uvs"]
)
def test_buffer_mesh_texcoords(mode_mono, texcoords):
    mesh = BufferMeshShape(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [1, 2, 3]],
        texcoords=texcoords,
    )
    mi_mesh = mesh.instance
    mi_params = mi.traverse(mi_mesh)
    assert "vertex_texcoords" in mi_params

    if texcoords is None:
        assert len(mi_params["vertex_texcoords"]) == 0
    else:
        np.testing.assert_array_equal(
            np.array(mi_params["vertex_texcoords"]), np.ravel(texcoords)
        )
