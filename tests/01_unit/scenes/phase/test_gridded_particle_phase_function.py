import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import GriddedParticlePhaseFunction
from eradiate.spectral.index import MonoSpectralIndex
from eradiate.testing.types import check_scene_element


def make_phase_data():
    """
    Build a dense, NaN-padded 2 r_eff by 1 v_eff mu/phase grid (one pair
    per row): the two pairs have different angle counts, on purpose, to
    exercise ragged compaction. r_eff entry 0 has phase matrix components
    [0, 10, 20, 30] over 3 angles, entry 1 has [100, 110, 120, 130] over 2
    angles (NaN-padded to match entry 0's 3).
    """
    mu = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 1.0, np.nan],
        ]
    )

    phase = np.full((2, 4, 3), np.nan)
    for ci, v in enumerate((0.0, 10.0, 20.0, 30.0)):
        phase[0, ci, :3] = v
    for ci, v in enumerate((100.0, 110.0, 120.0, 130.0)):
        phase[1, ci, :2] = v

    ext = ureg.Quantity(np.array([[2.0], [3.0]]), "1/km")
    ssa = ureg.Quantity(np.array([[0.5], [0.6]]), "dimensionless")

    return {"mu": mu, "phase": phase, "ext": ext, "ssa": ssa, "max_union_size": 5}


def make_particle_phase(r_eff_volume=None, v_eff_volume=None, **kwargs):
    """Build a minimal 2-reff by 1-veff :class:`.GriddedParticlePhaseFunction`."""
    # n_x_cells != n_z_cells on purpose: r_eff_volume/v_eff_volume must be
    # shaped (n_x_cells, n_y_cells, n_z_cells), matching GridCoords.shape.
    # With n_x_cells == n_z_cells (as a 1x1xN grid would give), a wrongly
    # transposed array is still broadcast-compatible and the mismatch goes
    # undetected; making them different forces the exact-shape check in
    # _prepare_grid() to actually run.
    grid = CartesianGrid(
        x_edges=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        y_edges=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    if r_eff_volume is None:
        r_eff_volume = ureg.Quantity(
            np.full((grid.n_x_cells, grid.n_y_cells, grid.n_z_cells), 10.0), "micron"
        )
    if v_eff_volume is None:
        v_eff_volume = ureg.Quantity(
            np.full((grid.n_x_cells, grid.n_y_cells, grid.n_z_cells), 0.1),
            "dimensionless",
        )
    phase_data_dict = make_phase_data()

    kwargs.setdefault("grid_1", ureg.Quantity(np.array([10.0, 20.0]), "micron"))
    kwargs.setdefault("grid_2", ureg.Quantity(np.array([0.1]), "dimensionless"))

    return GriddedParticlePhaseFunction(
        geometry=geometry,
        volume_1=r_eff_volume,
        volume_2=v_eff_volume,
        phase_data=lambda si: phase_data_dict,
        **kwargs,
    )


class TestGriddedParticlePhaseFunction:
    def test_kernel_dict(self, mode_mono):
        """The kernel template/params expose the expected type, fields and grids."""
        phase = make_particle_phase()
        check_scene_element(phase, mi.PhaseFunction)

        template, params = traverse(phase)
        assert template["type"] == "particlephase"
        np.testing.assert_allclose(
            np.array(template["r_eff_grid"]).ravel(), [10.0, 20.0]
        )
        np.testing.assert_allclose(np.array(template["v_eff_grid"]).ravel(), [0.1])

        assert set(params.keys()) == {
            "nodes",
            "phase_mueller",
            "grid_start",
            "sigma_s_weight",
        }

    def test_build_phase_parameters(self):
        """Ragged (reff, veff) phase data is correctly compacted and expanded to Mueller form."""
        phase = make_particle_phase()
        ctx = KernelContext(si=MonoSpectralIndex(w=ureg.Quantity(1.0, "micron")))

        built = phase._build_phase_parameters(ctx.si)

        np.testing.assert_allclose(
            np.array(built["nodes"]).ravel(), [-1.0, 0.0, 1.0, -1.0, 1.0]
        )
        np.testing.assert_allclose(np.array(built["grid_start"]).ravel(), [0, 3, 5])
        np.testing.assert_allclose(
            np.array(built["sigma_s_weight"]).ravel(), [1.0, 1.8]
        )

        # spherical particle_shape: m11=comp0, m12=comp1, m22=comp0, m33=comp2,
        # m34=comp3, m44=comp2; r_eff entry 0 has comps [0,10,20,30] over 3
        # angles, entry 1 has [100,110,120,130] over 2 angles.
        expected_mueller_row0 = [0.0, 10.0, 0.0, 20.0, 30.0, 20.0]
        expected_mueller_row1 = [100.0, 110.0, 100.0, 120.0, 130.0, 120.0]
        expected_mueller = np.array(
            [expected_mueller_row0] * 3 + [expected_mueller_row1] * 2
        )
        np.testing.assert_allclose(
            np.array(built["phase_mueller"]).ravel(), expected_mueller.flatten()
        )

    def test_r_eff_volume_wrong_axis_order_raises(self, mode_mono):
        """r_eff_volume/v_eff_volume must be (n_x_cells, n_y_cells, n_z_cells);
        a (n_z_cells, n_y_cells, n_x_cells) array must be rejected, not silently
        broadcast."""
        grid = CartesianGrid(
            x_edges=np.array([-1.0, 0.0, 1.0]) * ureg.km,
            y_edges=np.array([-1.0, 1.0]) * ureg.km,
            levels=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
        )
        wrong_order = ureg.Quantity(
            np.full((grid.n_z_cells, grid.n_y_cells, grid.n_x_cells), 10.0), "micron"
        )
        phase = make_particle_phase(r_eff_volume=wrong_order)

        with pytest.raises(ValueError, match="Invalid grid shape"):
            check_scene_element(phase, mi.PhaseFunction)

    @pytest.mark.parametrize(
        "grid, match",
        [
            (np.array([[10.0, 20.0]]), "1-D array"),
            (np.array([20.0, 10.0]), "strictly increasing"),
        ],
    )
    def test_r_eff_grid_validation(self, grid, match):
        """r_eff_grid must be a strictly increasing 1-D array."""
        with pytest.raises(ValueError, match=match):
            make_particle_phase(grid_1=grid)

    @pytest.mark.parametrize(
        "grid, match",
        [
            (np.array([20.0, 10.0]), "strictly increasing"),
        ],
    )
    def test_v_eff_grid_validation(self, grid, match):
        """v_eff_grid must be a strictly increasing 1-D array."""
        with pytest.raises(ValueError, match=match):
            make_particle_phase(grid_2=grid)

    def test_r_eff_grid_irregular_accepted(self):
        """r_eff_grid may be strictly increasing without being regularly spaced."""
        make_particle_phase(
            grid_1=ureg.Quantity(np.array([10.0, 15.0, 30.0]), "micron")
        )

    def test_v_eff_grid_irregular_accepted(self):
        """v_eff_grid may be strictly increasing without being regularly spaced."""
        make_particle_phase(
            grid_2=ureg.Quantity(np.array([0.1, 0.15, 0.3]), "dimensionless")
        )
