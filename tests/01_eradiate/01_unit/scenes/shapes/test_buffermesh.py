import mitsuba as mi
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
