import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.radprops import ZGrid


@pytest.mark.parametrize(
    "levels",
    [
        np.linspace(0, 100, 11) * ureg.km,
        np.linspace(0, 100000, 11),
    ],
    ids=["pint", "unitless"],
)
def test_zgrid_all(levels):
    zgrid = ZGrid(levels=levels)

    np.testing.assert_array_equal(
        zgrid.levels.m_as("km"), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    assert zgrid.n_levels == 11

    np.testing.assert_array_equal(
        zgrid.layers.m_as("km"), [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    )
    assert zgrid.n_layers == 10

    assert zgrid.layer_height == 10 * ureg.km
    assert zgrid.total_height == 100 * ureg.km


def test_zgrid_fails():
    with pytest.raises(ValueError, match="levels must be regularly spaced"):
        ZGrid([0, 1, 3])
