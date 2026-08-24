import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.grid import GridCoords


@pytest.mark.parametrize(
    "levels",
    [
        np.linspace(0, 100, 11) * ureg.km,
        np.linspace(0, 100000, 11),
    ],
    ids=["pint", "unitless"],
)
def test_grid_all(levels):
    grid = GridCoords.convert(levels)

    np.testing.assert_array_equal(
        grid.levels.m_as("km"), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    assert grid.n_levels == 11

    np.testing.assert_array_equal(
        grid.layers.m_as("km"), [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    )
    assert grid.n_layers == 10

    assert grid.layer_height == 10 * ureg.km
    assert grid.total_height == 100 * ureg.km


def test_grid_fails():
    with pytest.raises(ValueError, match="levels must be regularly spaced"):
        GridCoords.convert(np.asarray([0, 1, 3]))
