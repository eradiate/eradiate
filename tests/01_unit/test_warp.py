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
