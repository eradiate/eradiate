import mitsuba as mi
import pytest

from eradiate.scenes.shapes import BufferMeshShape


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
