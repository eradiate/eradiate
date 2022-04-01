import numpy as np
import pytest

from eradiate.quad import Quad


@pytest.mark.parametrize(
    "n, exp_nodes, exp_weights",
    [
        (
            1,
            [0],
            [2],
        ),
        (
            2,
            [-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)],
            [1, 1],
        ),
        (
            3,
            [-np.sqrt(3.0 / 5.0), 0, np.sqrt(3.0 / 5.0)],
            [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
        ),
        (
            4,
            [-0.861136, -0.339981, 0.339981, 0.861136],
            [0.347855, 0.652145, 0.652145, 0.347855],
        ),
    ],
    ids=list([f"n-{i}" for i in range(1, 5)]),
)
def test_gauss_legendre(n, exp_nodes, exp_weights, mode_mono):
    quad = Quad.gauss_legendre(n)
    assert np.allclose(quad.nodes, exp_nodes)
    assert np.allclose(quad.weights, exp_weights)


@pytest.mark.parametrize(
    "n, exp_nodes, exp_weights",
    [
        (
            2,
            [-1, 1],
            [1.0, 1.0],
        ),
        (
            3,
            [-1, 0, 1],
            [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
        ),
        (
            4,
            [-1, -np.sqrt(1.0 / 5.0), np.sqrt(1.0 / 5.0), 1],
            [1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0],
        ),
        (
            5,
            [-1, -np.sqrt(3.0 / 7.0), 0, np.sqrt(3.0 / 7.0), 1],
            [1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0],
        ),
    ],
    ids=list([f"n-{i}" for i in range(2, 6)]),
)
def test_gauss_lobatto(n, exp_nodes, exp_weights, mode_mono):
    quad = Quad.gauss_lobatto(n)
    assert np.allclose(quad.nodes, exp_nodes)
    assert np.allclose(quad.weights, exp_weights)


def test_quad_integrate(modes_all):
    quad = Quad.gauss_legendre(10)
    f = lambda x: x**2
    values = np.array([f(x) for x in quad.eval_nodes(interval=(0, 1))])
    assert np.allclose(quad.integrate(values, interval=(0, 1)), 1.0 / 3.0)
