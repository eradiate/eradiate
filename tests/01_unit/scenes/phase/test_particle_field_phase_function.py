import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import ParticleFieldPhaseFunction
from eradiate.spectral.index import MonoSpectralIndex
from eradiate.test_tools.types import check_scene_element


def make_phase_data():
    """
    Build mu and phase arrays for a 2 r_eff by 1 v_eff grid, nan padded to
    the largest pair. The two pairs have different angle counts, on
    purpose, to exercise ragged compaction. r_eff entry 0 has phase matrix
    components [0, 10, 20, 30], entry 1 has [100, 110, 120, 130], each
    constant across its own angles.
    """
    mu = np.full((2, 1, 3), np.nan)
    mu[0, 0, :] = [-1.0, 0.0, 1.0]
    mu[1, 0, :2] = [-1.0, 1.0]

    phase = np.full((4, 2, 1, 3), np.nan)
    for ci, v in enumerate((0.0, 10.0, 20.0, 30.0)):
        phase[ci, 0, 0, :] = v
    for ci, v in enumerate((100.0, 110.0, 120.0, 130.0)):
        phase[ci, 1, 0, :2] = v

    nangles = np.array([[3], [2]])
    ext = np.array([[2.0], [3.0]])
    ssa = np.array([[0.5], [0.6]])

    return mu, phase, nangles, ext, ssa


def make_particle_phase(r_eff_volume=None, v_eff_volume=None, **kwargs):
    """Build a minimal 2-reff by 1-veff :class:`.ParticleFieldPhaseFunction`."""
    # n_cells_x != n_cells_z on purpose: r_eff_volume/v_eff_volume must be
    # shaped (n_cells_x, n_cells_y, n_cells_z), matching GridCoords.shape.
    # With n_cells_x == n_cells_z (as a 1x1xN grid would give), a wrongly
    # transposed array is still broadcast-compatible and the mismatch goes
    # undetected; making them different forces the exact-shape check in
    # _prepare_grid() to actually run.
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    if r_eff_volume is None:
        r_eff_volume = ureg.Quantity(
            np.full((grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), 10.0), "micron"
        )
    if v_eff_volume is None:
        v_eff_volume = ureg.Quantity(
            np.full((grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), 0.1),
            "dimensionless",
        )
    phase_data_tuple = make_phase_data()

    kwargs.setdefault("r_eff_grid", np.array([10.0, 20.0]))
    kwargs.setdefault("v_eff_grid", np.array([0.1]))

    return ParticleFieldPhaseFunction(
        geometry=geometry,
        r_eff_volume=lambda ctx: r_eff_volume,
        v_eff_volume=lambda ctx: v_eff_volume,
        phase_data=lambda si: phase_data_tuple,
        **kwargs,
    )


def test_particle_phase_kernel_dict(mode_mono):
    """The kernel template/params expose the expected type, fields and grids."""
    phase = make_particle_phase()
    check_scene_element(phase, mi.PhaseFunction)

    template, params = traverse(phase)
    assert template["type"] == "particlefieldphase"
    assert template["blending_method"] == "search"
    np.testing.assert_allclose(np.array(template["r_eff_grid"]).ravel(), [10.0, 20.0])
    np.testing.assert_allclose(np.array(template["v_eff_grid"]).ravel(), [0.1])

    assert set(params.keys()) == {
        "nodes",
        "phase_mueller",
        "grid_start",
        "grid_len",
        "sigma_s_weight",
    }


def test_particle_phase_build_phase_parameters():
    """Ragged (reff, veff) phase data is correctly compacted and expanded to Mueller form."""
    phase = make_particle_phase()
    ctx = KernelContext(si=MonoSpectralIndex(w=ureg.Quantity(1.0, "micron")))

    built = phase._build_phase_parameters(ctx.si)

    np.testing.assert_allclose(np.array(built["nodes"]), [-1.0, 0.0, 1.0, -1.0, 1.0])
    np.testing.assert_allclose(np.array(built["grid_start"]), [0, 3])
    np.testing.assert_allclose(np.array(built["grid_len"]), [3, 2])
    np.testing.assert_allclose(np.array(built["sigma_s_weight"]), [1.0, 1.8])

    # spherical particle_shape: m11=comp0, m12=comp1, m22=comp0, m33=comp2,
    # m34=comp3, m44=comp2; r_eff entry 0 has comps [0,10,20,30] over 3
    # angles, entry 1 has [100,110,120,130] over 2 angles.
    expected_mueller_row0 = [0.0, 10.0, 0.0, 20.0, 30.0, 20.0]
    expected_mueller_row1 = [100.0, 110.0, 100.0, 120.0, 130.0, 120.0]
    expected_mueller = np.array(
        [expected_mueller_row0] * 3 + [expected_mueller_row1] * 2
    )
    np.testing.assert_allclose(
        np.array(built["phase_mueller"]), expected_mueller.flatten()
    )


def test_particle_phase_sigma_s_override():
    """sigma_s_override replaces the per-pair ext*ssa scattering weight everywhere."""
    phase = make_particle_phase(sigma_s_override=42.0)
    ctx = KernelContext(si=MonoSpectralIndex(w=ureg.Quantity(1.0, "micron")))

    built = phase._build_phase_parameters(ctx.si)
    np.testing.assert_allclose(np.array(built["sigma_s_weight"]), [42.0, 42.0])


def test_particle_phase_r_eff_volume_wrong_axis_order_raises(mode_mono):
    """r_eff_volume/v_eff_volume must be (n_cells_x, n_cells_y, n_cells_z);
    a (n_cells_z, n_cells_y, n_cells_x) array must be rejected, not silently
    broadcast."""
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
    )
    wrong_order = ureg.Quantity(
        np.full((grid.n_cells_z, grid.n_cells_y, grid.n_cells_x), 10.0), "micron"
    )
    phase = make_particle_phase(r_eff_volume=wrong_order)

    with pytest.raises(ValueError, match="Invalid grid shape"):
        check_scene_element(phase, mi.PhaseFunction)


@pytest.mark.parametrize(
    "grid, match",
    [
        (np.array([[10.0, 20.0]]), "1-D array"),
        (np.array([20.0, 10.0]), "strictly increasing"),
        (np.array([10.0, 15.0, 30.0]), "regularly spaced"),
    ],
)
def test_particle_phase_r_eff_grid_validation(grid, match):
    """r_eff_grid must be a strictly increasing, regularly-spaced 1-D array."""
    with pytest.raises(ValueError, match=match):
        make_particle_phase(r_eff_grid=grid)


@pytest.mark.parametrize(
    "grid, match",
    [
        (np.array([20.0, 10.0]), "strictly increasing"),
        (np.array([0.1, 0.15, 0.3]), "regularly spaced"),
    ],
)
def test_particle_phase_v_eff_grid_validation(grid, match):
    """v_eff_grid must be a strictly increasing, regularly-spaced 1-D array."""
    with pytest.raises(ValueError, match=match):
        make_particle_phase(v_eff_grid=grid)
