import mitsuba as mi
import numpy as np

from eradiate import warp


def check_inverse(func, inverse):
    for x in np.linspace(1e-6, 1 - 1e-6, 10):
        for y in np.linspace(1e-6, 1 - 1e-6, 10):
            p1 = np.array([x, y])
            p2 = func(p1)
            p3 = inverse(p2)
            assert np.allclose(p1, p3, atol=1e-5)


def test_square_to_uniform_disk_concentric():
    assert np.allclose(
        warp.square_to_uniform_disk_concentric([0, 0]),
        ([-1 / np.sqrt(2), -1 / np.sqrt(2)]),
    )
    assert np.allclose(warp.square_to_uniform_disk_concentric([0.5, 0.5]), [0, 0])

    check_inverse(
        warp.square_to_uniform_disk_concentric, warp.uniform_disk_to_square_concentric
    )


def test_square_to_uniform_hemisphere():
    assert np.allclose(warp.square_to_uniform_hemisphere([0.5, 0.5]), [0, 0, 1])
    assert np.allclose(warp.square_to_uniform_hemisphere([0, 0.5]), [-1, 0, 0])

    check_inverse(warp.square_to_uniform_hemisphere, warp.uniform_hemisphere_to_square)
    # check_vectorization("square_to_uniform_hemisphere")


def test_transform_affine(mode_mono):
    trafo_scale = mi.Transform4f.scale((2, 3, 4))
    trafo_translate = mi.Transform4f.translate((0.1, 0.2, 0.3))
    trafo_rotate = mi.Transform4f.rotate((0, 0, 1), 90)

    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert np.allclose(
        warp.transform_affine(trafo_scale, points),
        np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]),
    )
    assert np.allclose(
        warp.transform_affine(trafo_translate, points),
        np.array([[1.1, 0.2, 0.3], [0.1, 1.2, 0.3], [0.1, 0.2, 1.3]]),
    )
    assert np.allclose(
        warp.transform_affine(trafo_rotate, points),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    )
