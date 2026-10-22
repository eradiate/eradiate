import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.experiments._helpers import (
    check_piecewise_compatible,
    measure_inside_atmosphere,
    surface_converter,
)
from eradiate.grid import CartesianGrid
from eradiate.scenes.atmosphere import HeterogeneousAtmosphere
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.surface import BasicSurface


def test_helpers_measure_inside_atmosphere(
    mode_mono,
    atmosphere_us_standard_mono,
):
    atm = HeterogeneousAtmosphere(
        geometry="plane_parallel",
        molecular_atmosphere=atmosphere_us_standard_mono,
    )

    s1 = eradiate.scenes.measure.MultiDistantMeasure()
    assert not measure_inside_atmosphere(atm, s1)

    s2 = eradiate.scenes.measure.PerspectiveCameraMeasure(
        origin=[1, 1, 1], target=[0, 0, 0]
    )
    assert measure_inside_atmosphere(atm, s2)

    s3 = eradiate.scenes.measure.MultiRadiancemeterMeasure(
        origins=[[1, 1, 1], [0, 0, 1000000]], directions=[[0, 0, -1], [0, 0, -1]]
    )
    with pytest.raises(ValueError):
        measure_inside_atmosphere(atm, s3)


def test_helpers_surface_converter(mode_mono):
    # A dictionary specifying a surface is converted to a surface
    assert isinstance(surface_converter({"type": "basic"}), BasicSurface)

    # A dictionary specifying a BSDF is converted to a surface
    assert isinstance(surface_converter({"type": "lambertian"}), BasicSurface)


@pytest.mark.parametrize("n_x_cells, expected", [(1, True), (2, False)])
def test_helpers_check_piecewise_compatible_grid(
    mode_mono, atmosphere_us_standard_mono, n_x_cells, expected
):
    # The piecewise integrator is rejected for grids with horizontal resolution
    grid = CartesianGrid(
        x_edges=np.linspace(-1.0, 1.0, n_x_cells + 1) * ureg.km,
        y_edges=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.linspace(0.0, 120.0, 11) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid)
    atm = HeterogeneousAtmosphere(
        geometry=geometry, molecular_atmosphere=atmosphere_us_standard_mono
    )
    compatible, _ = check_piecewise_compatible(atm)
    assert compatible is expected
