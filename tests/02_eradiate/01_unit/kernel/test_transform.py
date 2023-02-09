import drjit as dr
import mitsuba as mi

from eradiate.kernel import map_cube, map_unit_cube


def test_map_unit_cube(mode_mono):
    """
    Returns a transformation that maps old cube vertices to new cube vertices.
    """
    trafo = map_unit_cube(1, 2, 3, 4, 5, 6)

    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    new_vertices = [
        [1, 3, 5],
        [2, 3, 5],
        [1, 4, 5],
        [2, 4, 5],
        [1, 3, 6],
        [2, 3, 6],
        [1, 4, 6],
        [2, 4, 6],
    ]

    for vertex, new_vertex in zip(vertices, new_vertices):
        assert dr.allclose(
            trafo.transform_affine(mi.Point3f(vertex)),
            new_vertex,
        )


def test_map_cube(mode_mono):
    """
    Returns a transformation that maps old cube vertices to new cube vertices.
    """

    trafo = map_cube(1, 2, 3, 4, 5, 6)

    vertices = [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ]

    new_vertices = [
        [1, 3, 5],
        [2, 3, 5],
        [1, 4, 5],
        [2, 4, 5],
        [1, 3, 6],
        [2, 3, 6],
        [1, 4, 6],
        [2, 4, 6],
    ]

    for vertex, new_vertex in zip(vertices, new_vertices):
        assert dr.allclose(
            trafo.transform_affine(mi.Point3f(vertex)),
            new_vertex,
        )
