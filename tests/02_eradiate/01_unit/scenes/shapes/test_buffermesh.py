import pytest

from eradiate.contexts import KernelDictContext
from eradiate.scenes.shapes import BufferMeshShape


def test_construct(modes_all_double):
    ctx = KernelDictContext()
    mesh = BufferMeshShape(
        vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], faces=[[1, 2, 3]]
    )
    assert mesh.kernel_dict(ctx=ctx)

    # flat list of vertex pos fails
    with pytest.raises(ValueError):
        mesh = BufferMeshShape(vertices=[1, 0, 0, 0, 1, 0, 0, 0, 1], faces=[[1, 2, 3]])

    # flat list of face indices fails
    with pytest.raises(ValueError):
        mesh = BufferMeshShape(
            vertices=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], faces=[1, 2, 3]
        )
