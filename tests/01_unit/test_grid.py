import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.grid import CartesianGrid, SpatialGrid, SphericalGrid
from eradiate.scenes._gridvolume import make_volume_grid
from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry


def _spherical_grid(azimuths_deg):
    return SphericalGrid(
        levels=np.array([0.0, 1.0]) * ureg.km,
        azimuths=np.asarray(azimuths_deg) * ureg.deg,
        colatitudes=np.array([0.0, 180.0]) * ureg.deg,
    )


class TestSpatialGrid:
    @pytest.mark.parametrize(
        "value, expected_type, expected_n_layers",
        [
            ("cartesian", CartesianGrid, 1200),
            (np.linspace(0.0, 1000.0, 11), CartesianGrid, 10),
            ([0.0, 500.0, 1000.0], CartesianGrid, 2),
            (np.linspace(0.0, 1.0, 11) * ureg.km, CartesianGrid, 10),
            (
                {
                    "type": "spherical",
                    "levels": [0.0, 1.0] * ureg.km,
                    "azimuths": [0.0, 360.0] * ureg.deg,
                    "colatitudes": [0.0, 180.0] * ureg.deg,
                },
                SphericalGrid,
                1,
            ),
        ],
        ids=["str", "ndarray", "list", "quantity", "dict"],
    )
    def test_convert(self, value, expected_type, expected_n_layers):
        grid = SpatialGrid.convert(value)
        assert isinstance(grid, expected_type)
        assert grid.n_layers == expected_n_layers

    @pytest.mark.parametrize(
        "value", ["spherical", {"type": "foo"}], ids=["str", "dict"]
    )
    def test_convert_invalid(self, value):
        with pytest.raises(ValueError):
            SpatialGrid.convert(value)

    @pytest.mark.parametrize(
        "geometry_cls, xy_edges, expected_type",
        [
            (PlaneParallelGeometry, [-1.0, 0.0, 1.0] * ureg.km, CartesianGrid),
            (SphericalShellGeometry, [0.0, 90.0, 180.0] * ureg.deg, SphericalGrid),
        ],
        ids=["plane_parallel", "spherical_shell"],
    )
    def test_convert_geometry(self, geometry_cls, xy_edges, expected_type):
        levels = np.linspace(0.0, 1000.0, 11) * ureg.m

        # 1D array: levels
        grid = geometry_cls(grid=levels).grid
        assert isinstance(grid, expected_type)
        assert grid.shape == (1, 1, 10)

        # Sequence of 3 arrays: (x, y, z) edges
        grid = geometry_cls(grid=(xy_edges, xy_edges, levels)).grid
        assert isinstance(grid, expected_type)
        assert grid.shape == (2, 2, 10)

    @pytest.mark.parametrize("grid_cls", [CartesianGrid, SphericalGrid])
    def test_with_cell_counts(self, grid_cls):
        edges = [0.0, 1.0, 2.0, 3.0, 4.0]
        grid = grid_cls(edges, edges, edges)
        result = grid.with_cell_counts(n_x_cells=2, n_z_cells=8)
        assert type(result) is grid_cls
        assert result.shape == (2, 4, 8)
        for axis in "xyz":
            assert np.all(
                getattr(result, f"{axis}_edges")[[0, -1]]
                == getattr(grid, f"{axis}_edges")[[0, -1]]
            )

    def test_with_cell_sizes(self):
        grid = SphericalGrid(
            [0.0, 90.0] * ureg.deg, [0.0, 90.0] * ureg.deg, [0.0, 1.0] * ureg.km
        )
        result = grid.with_cell_sizes(
            x_cell_size=30 * ureg.deg, z_cell_size=250 * ureg.m
        )
        assert result.shape == (3, 1, 4)

        with pytest.raises(ValueError, match="tolerance"):
            grid.with_cell_sizes(x_cell_size=40 * ureg.deg, x_atol=1 * ureg.deg)


class TestCartesianGrid:
    @pytest.mark.parametrize(
        "x_edges", [[0.0, 1.0, 3.0], [0.0]], ids=["irregular", "too_short"]
    )
    def test_invalid_edges(self, x_edges):
        with pytest.raises(ValueError, match="x_edges"):
            CartesianGrid(levels=[0.0, 1.0], x_edges=x_edges, y_edges=[0.0, 1.0])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"x_edges": [0, 1], "y_edges": [0, 1]}, "z_edges"),
            (
                {
                    "x_edges": [0, 1],
                    "y_edges": [0, 1],
                    "z_edges": [0, 1],
                    "levels": [0, 1],
                },
                "levels",
            ),
            ({"y_edges": [0, 1], "levels": [0, 1]}, "x_edges"),
        ],
        ids=["missing", "both", "missing_x"],
    )
    def test_alias_exclusive(self, kwargs, match):
        with pytest.raises(TypeError, match=match):
            CartesianGrid(**kwargs)


class TestSphericalGrid:
    def test_aliases(self):
        """Aliases and generic arguments produce the same storage."""
        edges = {
            "x": [0.0, 90.0, 180.0] * ureg.deg,
            "y": [0.0, 180.0] * ureg.deg,
            "z": [0.0, 1.0, 2.0] * ureg.km,
        }
        generic = SphericalGrid(edges["x"], edges["y"], edges["z"])
        aliased = SphericalGrid(
            azimuths=edges["x"], colatitudes=edges["y"], levels=edges["z"]
        )
        for axis in "xyz":
            assert np.all(
                getattr(generic, f"{axis}_edges") == getattr(aliased, f"{axis}_edges")
            )
        assert np.all(aliased.sectors == aliased.x_centers)
        assert aliased.band_width == aliased.y_cell_size

    @pytest.mark.parametrize(
        "azimuths, expected",
        [
            ([-180.0, 0.0, 180.0], (-180.0, 180.0, 0)),
            ([200.0, 225.0, 250.0], (-160.0, -110.0, 0)),
            ([0.0, 360.0], (-180.0, 180.0, 0)),
            (np.linspace(0.0, 360.0, 5), (-180.0, 180.0, 2)),
            (np.linspace(-90.0, 270.0, 5), (-180.0, 180.0, 3)),
        ],
        ids=["native", "shifted", "full_single_cell", "full_0_360", "full_m90_270"],
    )
    def test_kernel_azimuths(self, azimuths, expected):
        phi_min, phi_max, n_roll = _spherical_grid(azimuths).kernel_azimuths()
        np.testing.assert_allclose(np.rad2deg([phi_min, phi_max]), expected[:2])
        assert n_roll == expected[2]

    @pytest.mark.parametrize(
        "azimuths",
        [[170.0, 190.0], np.linspace(0.0, 360.0, 4), [0.0, 400.0], [10.0, 0.0]],
        ids=["partial_across_cut", "cut_inside_cell", "too_wide", "decreasing"],
    )
    def test_kernel_azimuths_invalid(self, azimuths):
        with pytest.raises(ValueError):
            _spherical_grid(azimuths)

    def test_to_world(self, mode_mono):
        """to_world maps the azimuth range to its place in [0, 1]."""
        to_world = _spherical_grid([200.0, 225.0, 250.0]).to_world
        z = [(to_world @ [0.0, 0.0, w])[2] for w in (0.0, 1.0)]
        np.testing.assert_allclose(z, np.array([-160.0, -110.0]) / 360.0 + 0.5)

    def test_make_volume_grid_roll(self, mode_mono):
        """Volume data on [0°, 360°] is rolled to start at the -180° cell."""
        grid = _spherical_grid(np.linspace(0.0, 360.0, 5))
        geometry = SphericalShellGeometry(grid=grid)
        sectors = grid.sectors.m_as(ureg.deg)  # [45, 135, 225, 315]

        result = make_volume_grid(
            geometry,
            KernelContext(),
            eval_grid=lambda ctx: sectors.reshape(-1, 1, 1),
            dtype=np.float64,
            spectral_index=False,
            to_mi=False,
            extra_kwargs={},
        )
        np.testing.assert_array_equal(result[:, 0, 0], [225.0, 315.0, 45.0, 135.0])
