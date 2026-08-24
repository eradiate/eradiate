import numpy as np
import pytest
import xarray as xr

from eradiate.data.convert import make_aer_core_v2
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.atmosphere import ParticleEnsemble
from eradiate.scenes.atmosphere._particle_field import PHAMAT_LABELS, ParticleField
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import ParticleFieldPhaseFunction
from eradiate.spectral.index import MonoSpectralIndex
from eradiate.units import unit_registry as ureg


def make_properties_dataset(
    reff=10.0,
    veff=0.1,
    w=(0.5, 1.5),
    n_theta=5,
    nphamat=4,
    phase_value=1.0,
    ext_value=1.0,
    ssa_value=0.8,
) -> xr.Dataset:
    """
    Build a Prt v1 properties dataset (with ``reff``/``veff`` dimensions)
    directly, with a shared ascending mu grid across the whole
    ``(w, reff, veff)`` grid.
    """
    reff_values = np.atleast_1d(np.asarray(reff, dtype=float))
    veff_values = np.atleast_1d(np.asarray(veff, dtype=float))
    w_values = np.atleast_1d(np.asarray(w, dtype=float))
    nw, nreff, nveff = len(w_values), len(reff_values), len(veff_values)

    mu_1d = np.linspace(-1.0, 1.0, n_theta)
    mu = np.tile(mu_1d, (nw, nreff, nveff, 1))
    theta = np.degrees(np.arccos(mu))

    ext = np.broadcast_to(np.asarray(ext_value, dtype=float), (nw, nreff, nveff)).copy()
    ssa = np.broadcast_to(np.asarray(ssa_value, dtype=float), (nw, nreff, nveff)).copy()
    phase = np.broadcast_to(
        np.asarray(phase_value, dtype=float), (nphamat, nw, nreff, nveff, n_theta)
    ).copy()

    return make_aer_core_v2(
        w=w_values * ureg.micron,
        phamat=PHAMAT_LABELS[nphamat],
        mu=mu * ureg.dimensionless,
        theta=theta * ureg.degree,
        ext=ext * ureg("km^-1 / (g / m^3)"),
        ssa=ssa * ureg.dimensionless,
        phase=phase * ureg("1/sr"),
        reff=reff_values * ureg.micron,
        veff=veff_values * ureg.dimensionless,
    )


def make_profile_dataset(
    grid: PlaneParallelGridCoords,
    reff=10.0,
    veff=0.1,
    mass_concentration=1e-3,
    z_levels=None,
) -> xr.Dataset:
    """Build a Ppr v1 profile dataset densely filling every voxel of ``grid``."""
    n_x, n_y = grid.n_cells_x, grid.n_cells_y
    z_levels_km = (
        np.asarray(z_levels, dtype=float)
        if z_levels is not None
        else grid.levels.m_as("km")
    )
    n_z = len(z_levels_km) - 1

    ix, iy, iz = (
        a.ravel()
        for a in np.meshgrid(
            np.arange(n_x), np.arange(n_y), np.arange(n_z), indexing="ij"
        )
    )
    n_pts = n_x * n_y * n_z

    data_vars = {
        "i_x": (["index"], ix),
        "i_y": (["index"], iy),
        "i_z": (["index"], iz),
        "reff": (
            ["index"],
            np.broadcast_to(np.asarray(reff, dtype=float), n_pts),
            {"units": "micron"},
        ),
        "veff": (
            ["index"],
            np.broadcast_to(np.asarray(veff, dtype=float), n_pts),
            {"units": "dimensionless"},
        ),
        "mass_concentration": (
            ["index"],
            np.broadcast_to(np.asarray(mass_concentration, dtype=float), n_pts),
            {"units": "g/m^3"},
        ),
    }
    coords = {
        "x_levels": (["x"], grid.edges_x.m_as("km"), {"units": "kilometer"}),
        "y_levels": (["y"], grid.edges_y.m_as("km"), {"units": "kilometer"}),
        "z_levels": (["z"], z_levels_km, {"units": "kilometer"}),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)


def make_particle_field(
    n_x: int = 2,
    n_y: int = 2,
    n_z: int = 3,
    x_extent: ureg.Quantity = 2.0 * ureg.km,
    y_extent: ureg.Quantity = 2.0 * ureg.km,
    z_extent: ureg.Quantity = 3.0 * ureg.km,
    reff=10.0,
    veff=0.1,
    mass_concentration=1e-3,
    w=(0.5, 1.5),
    ext_value=1.0,
    ssa_value=0.8,
    has_absorption=True,
    has_scattering=True,
) -> ParticleField:
    """Build a dense, spatially-uniform :class:`.ParticleField`."""
    grid = PlaneParallelGridCoords(
        edges_x=np.linspace(0.0, x_extent.m_as("km"), n_x + 1) * ureg.km,
        edges_y=np.linspace(0.0, y_extent.m_as("km"), n_y + 1) * ureg.km,
        levels=np.linspace(0.0, z_extent.m_as("km"), n_z + 1) * ureg.km,
    ).centered()
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])

    return ParticleField(
        profile=make_profile_dataset(
            grid, reff=reff, veff=veff, mass_concentration=mass_concentration
        ),
        properties=make_properties_dataset(
            reff=reff, veff=veff, w=w, ext_value=ext_value, ssa_value=ssa_value
        ),
        geometry=geometry,
        has_absorption=has_absorption,
        has_scattering=has_scattering,
    )


def _make_z_profile_particle_field(z_levels_src, target_levels):
    """Build a single-reff/veff :class:`.ParticleField` with a z-varying
    profile, resampled onto ``target_levels``."""
    z_levels_src = np.asarray(z_levels_src, dtype=float)
    z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])

    target_grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.asarray(target_levels, dtype=float) * ureg.km,
    )
    geometry = PlaneParallelGeometry(
        grid=target_grid,
        toa_altitude=target_grid.levels[-1],
        ground_altitude=target_grid.levels[0],
    )
    profile = make_profile_dataset(
        target_grid,
        reff=10.0 + z_centers_src,
        veff=0.1,
        mass_concentration=z_centers_src,
        z_levels=z_levels_src,
    )
    # Properties has a single reff/veff point, so ext/ssa interpolation is
    # constant regardless of the profile's own per-voxel reff, which is
    # therefore free to vary with z here.
    pf = ParticleField(
        profile=profile, properties=make_properties_dataset(), geometry=geometry
    )
    return pf, target_grid


def make_particle_ensemble(
    w_ref=1.0 * ureg.micron,
    sigma_t_val=1.0,
    ssa_val=0.6,
    tau_ref=1.0,
    n_mu=5,
    bottom=0.0 * ureg.km,
    top=2.0 * ureg.km,
) -> ParticleEnsemble:
    """Build a minimal single-phamat, spatially-uniform :class:`.ParticleEnsemble`."""
    mu = np.linspace(-1.0, 1.0, n_mu)
    theta = np.rad2deg(np.arccos(mu))

    pp_ds = make_aer_core_v2(
        w=ureg.Quantity([w_ref.m_as("micron")], "micron"),
        phamat=["11"],
        mu=ureg.Quantity(mu[np.newaxis, :], "dimensionless"),
        theta=ureg.Quantity(theta[np.newaxis, :], "degree"),
        ext=ureg.Quantity([sigma_t_val], "1/km"),
        ssa=ureg.Quantity([ssa_val], "dimensionless"),
        phase=ureg.Quantity(np.ones((1, 1, n_mu)), "1/sr"),
        pmom=np.zeros((1, 1)),
    )

    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([bottom.m_as("km"), top.m_as("km")]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])

    return ParticleEnsemble(
        bottom=bottom,
        top=top,
        distribution="uniform",
        w_ref=w_ref,
        tau_ref=ureg.Quantity(tau_ref, "dimensionless"),
        particle_properties=pp_ds,
        geometry=geometry,
    )


def test_particle_field_construct_basic():
    """A ParticleField can be constructed from a profile and a properties dataset."""
    assert isinstance(make_particle_field(), ParticleField)


@pytest.mark.parametrize(
    "z_levels",
    [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.5, 4.0]],
    ids=["regular", "irregular"],
)
def test_profile_regular_grid(z_levels):
    """_profile_regular_grid halves the profile's smallest z-gap and preserves x/y extent."""
    grid = PlaneParallelGridCoords(
        edges_x=np.array([0.0, 2.0, 4.0]) * ureg.km,
        edges_y=np.array([1.0, 3.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    profile = make_profile_dataset(grid, z_levels=z_levels)
    pf = ParticleField(
        profile=profile, properties=make_properties_dataset(), geometry=geometry
    )

    reg_grid = pf._profile_regular_grid()

    z_levels = np.asarray(z_levels)
    min_gap = np.diff(z_levels).min()
    levels_km = reg_grid.levels.m_as("km")

    np.testing.assert_allclose(levels_km[0], z_levels[0])
    np.testing.assert_allclose(levels_km[-1], z_levels[-1])
    np.testing.assert_allclose(np.diff(levels_km), min_gap / 2.0)

    assert reg_grid.n_cells_x == grid.n_cells_x
    assert reg_grid.n_cells_y == grid.n_cells_y
    edges_x_km = reg_grid.edges_x.m_as("km")
    np.testing.assert_allclose(edges_x_km[0], -edges_x_km[-1])


@pytest.mark.parametrize(
    "has_absorption, has_scattering, expected",
    [
        (True, True, {"sigma_t": 1.0, "albedo": 0.6, "sigma_s": 0.6, "sigma_a": 0.4}),
        (True, False, {"sigma_t": 1.0, "albedo": 0.0, "sigma_s": 0.0, "sigma_a": 1.0}),
        (False, True, {"sigma_t": 1.0, "albedo": 1.0, "sigma_s": 1.0, "sigma_a": 0.0}),
    ],
    ids=["absorbing_and_scattering", "absorbing_only", "scattering_only"],
)
def test_eval_family_values(has_absorption, has_scattering, expected):
    """sigma_t/albedo/sigma_s/sigma_a/mfp are consistent under all has_absorption/has_scattering combinations."""
    pf = make_particle_field(
        w=(1.0,),
        mass_concentration=0.5,
        ext_value=2.0,
        ssa_value=0.6,
        has_absorption=has_absorption,
        has_scattering=has_scattering,
    )
    si = MonoSpectralIndex(w=ureg.Quantity(1.0, "micron"))
    shape = (
        pf.geometry.grid.n_cells_x,
        pf.geometry.grid.n_cells_y,
        pf.geometry.grid.n_cells_z,
    )

    sigma_t = pf.eval_sigma_t(si)
    assert sigma_t.units == ureg.Unit("1/km")
    assert sigma_t.shape == shape
    np.testing.assert_allclose(sigma_t.m_as("1/km"), expected["sigma_t"])

    albedo = pf.eval_albedo(si)
    assert albedo.units == ureg.dimensionless
    np.testing.assert_allclose(albedo.m_as("dimensionless"), expected["albedo"])

    sigma_s = pf.eval_sigma_s(si)
    assert sigma_s.units == ureg.Unit("1/km")
    assert sigma_s.shape == shape
    np.testing.assert_allclose(sigma_s.m_as("1/km"), expected["sigma_s"])

    sigma_a = pf.eval_sigma_a(si)
    assert sigma_a.units == ureg.Unit("1/km")
    assert sigma_a.shape == shape
    np.testing.assert_allclose(sigma_a.m_as("1/km"), expected["sigma_a"])

    mfp = pf.eval_mfp(si)
    assert mfp.units == ureg.Unit("m")
    assert mfp.shape == shape
    expected_mfp = (
        np.inf if expected["sigma_t"] == 0.0 else 1.0 / expected["sigma_t"] * 1000.0
    )
    np.testing.assert_allclose(mfp.m_as("m"), expected_mfp)


def test_eval_family_no_absorption_no_scattering():
    """sigma_t/albedo/mfp raise, but sigma_s/sigma_a return zero, when both are disabled."""
    pf = make_particle_field(w=(1.0,), has_absorption=False, has_scattering=False)
    si = MonoSpectralIndex(w=ureg.Quantity(1.0, "micron"))

    with pytest.raises(RuntimeError, match="At least one"):
        pf.eval_sigma_t(si)
    with pytest.raises(RuntimeError, match="At least one"):
        pf.eval_albedo(si)
    with pytest.raises(RuntimeError, match="At least one"):
        pf.eval_mfp(si)

    np.testing.assert_allclose(pf.eval_sigma_s(si).m_as("1/km"), 0.0)
    np.testing.assert_allclose(pf.eval_sigma_a(si).m_as("1/km"), 0.0)


def test_eval_sigma_t_and_mfp_outside_cloud():
    """Voxels outside the profile's z-extent get zero sigma_t and infinite mfp."""
    pf, target_grid = _make_z_profile_particle_field(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], np.arange(-1.0, 6.1, 1.0)
    )
    si = MonoSpectralIndex(w=ureg.Quantity(1.0, "micron"))

    sigma_t = pf.eval_sigma_t(si, target_grid).m_as("1/km").ravel()
    mfp = pf.eval_mfp(si, target_grid).m_as("m").ravel()

    np.testing.assert_allclose(sigma_t[[0, -1]], 0.0)
    assert np.all(sigma_t[1:-1] > 0.0)
    assert np.all(np.isinf(mfp[[0, -1]]))
    assert np.all(np.isfinite(mfp[1:-1]))


def test_reff_veff_interpolation_and_clipping():
    """
    ext/ssa are interpolated continuously against each voxel's own
    reff/veff (dense, no ragged structure); voxels sitting outside the
    properties grid range are clipped rather than turning into NaN.
    """
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])

    # 2 populated voxels: one with reff inside the grid (interpolated),
    # one with reff far outside the grid range (must clip, not NaN).
    profile = xr.Dataset(
        data_vars={
            "i_x": (["index"], [0, 1]),
            "i_y": (["index"], [0, 0]),
            "i_z": (["index"], [0, 0]),
            "reff": (["index"], [15.0, 1000.0], {"units": "micron"}),
            "veff": (["index"], [0.1, 0.1], {"units": "dimensionless"}),
            "mass_concentration": (["index"], [1.0, 1.0], {"units": "g/m^3"}),
        },
        coords={
            "x_levels": (["x"], grid.edges_x.m_as("km"), {"units": "kilometer"}),
            "y_levels": (["y"], grid.edges_y.m_as("km"), {"units": "kilometer"}),
            "z_levels": (["z"], grid.levels.m_as("km"), {"units": "kilometer"}),
        },
    )
    properties = make_properties_dataset(
        reff=[10.0, 20.0],
        veff=0.1,
        w=(1.0,),
        ext_value=np.array([2.0, 4.0]).reshape(1, 2, 1),
    )

    pf = ParticleField(profile=profile, properties=properties, geometry=geometry)
    si = MonoSpectralIndex(w=ureg.Quantity(1.0, "micron"))

    sigma_t = pf.eval_sigma_t(si).m_as("1/km")
    assert np.all(np.isfinite(sigma_t))

    # Voxel 0 (reff=15, midway between 10 and 20): linearly interpolated.
    np.testing.assert_allclose(sigma_t[0, 0, 0], 3.0)
    # Voxel 1 (reff=1000, clipped to the grid's max of 20): same as reff=20.
    np.testing.assert_allclose(sigma_t[1, 0, 0], 4.0)


def test_phase_returns_particle_phase():
    """The phase property builds a ParticleFieldPhaseFunction matching the field's own grids."""
    pf = make_particle_field(w=(1.0,))

    phase = pf.phase

    assert isinstance(phase, ParticleFieldPhaseFunction)
    assert phase.blending_method == pf.phase_blending_method

    np.testing.assert_allclose(
        phase.r_eff_grid, pf.properties.reff.m_as("micron").astype(np.float64)
    )
    np.testing.assert_allclose(
        phase.v_eff_grid,
        pf.properties.veff.m_as(ureg.dimensionless).astype(np.float64),
    )


def test_phase_rebuilds_on_each_access():
    """Each access to phase builds a fresh ParticleFieldPhaseFunction instance."""
    pf = make_particle_field(w=(1.0,))
    assert pf.phase is not pf.phase


def test_phase_override_bypasses_construction():
    """Setting _phase short-circuits the phase property's normal construction."""
    pf = make_particle_field(w=(1.0,))
    sentinel = object()
    pf._phase = sentinel
    assert pf.phase is sentinel


def test_eval_phase_data_matches_grid_evals():
    """_eval_phase_data returns exactly _eval_phase_grid and _eval_ext_ssa_grid's own results, combined."""
    pf = make_particle_field(w=(1.0,))
    si = MonoSpectralIndex(w=ureg.Quantity(1.0, "micron"))

    mu, phase, nangles, ext, ssa = pf._eval_phase_data(si)
    expected_mu, expected_phase, expected_nangles = pf._eval_phase_grid(si)
    expected_ext, expected_ssa = pf._eval_ext_ssa_grid(si)

    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(phase, expected_phase)
    np.testing.assert_allclose(nangles, expected_nangles)
    np.testing.assert_allclose(ext, expected_ext.values)
    np.testing.assert_allclose(ssa, expected_ssa.values)


def test_template_and_params_phase_keys(mode_mono):
    """_template_phase/_params_phase expose the expected kernel dict keys."""
    pf = make_particle_field(w=(1.0,))

    template = pf._template_phase
    for key in [
        "type",
        "r_eff_volume.grid",
        "v_eff_volume.grid",
        "r_eff_grid",
        "v_eff_grid",
        "nodes",
        "phase_mueller",
        "grid_start",
        "grid_len",
        "blending_method",
    ]:
        assert key in template

    params = pf._params_phase
    for key in ["nodes", "phase_mueller", "grid_start", "grid_len"]:
        assert key in params


def test_to_profile_regular_grid():
    """to_profile_regular_grid returns a copy with a regularized geometry, sharing profile/properties."""
    pf = make_particle_field(w=(1.0,))

    result = pf.to_profile_regular_grid()

    assert result is not pf
    assert isinstance(result, ParticleField)
    expected_grid = pf._profile_regular_grid()
    np.testing.assert_allclose(
        result.geometry.grid.levels.m_as("km"), expected_grid.levels.m_as("km")
    )
    np.testing.assert_allclose(
        result.geometry.width.m_as("km"), pf.geometry.width.m_as("km")
    )
    assert result.profile is pf.profile
    assert result.properties is pf.properties


def test_to_particle_ensemble_from_profile_point_invalid_size_raises():
    """to_particle_ensemble_from_profile_point rejects a profile_point spanning more than one index."""
    pf = make_particle_field(w=(1.0,))
    resampled = pf.profile.resample_regular(pf.geometry.grid)

    with pytest.raises(ValueError, match="exactly one point"):
        ParticleField.to_particle_ensemble_from_profile_point(
            pf,
            resampled.data.isel(index=slice(0, 2)),
            w=ureg.Quantity(1.0, "micron"),
        )


def test_to_particle_ensemble_invalid_voxel_raises():
    """to_particle_ensemble raises when the requested voxel index has no cloud."""
    pf = make_particle_field(w=(1.0,))

    with pytest.raises(ValueError, match="No cloud voxel"):
        ParticleField.to_particle_ensemble(
            pf, w=ureg.Quantity(1.0, "micron"), ix=999, iy=999, iz=999
        )


def test_to_particle_ensemble_tau_ref_concentrated_at_voxel(mode_mono):
    """to_particle_ensemble places the whole voxel's tau_ref at its own (x, y) cell, zero elsewhere."""
    pf = make_particle_field(
        n_x=2,
        n_y=2,
        n_z=2,
        w=(1.0,),
        mass_concentration=0.5,
        ext_value=2.0,
        ssa_value=0.6,
    )
    grid = pf.geometry.grid
    w = ureg.Quantity(1.0, "micron")

    layer = ParticleField.to_particle_ensemble(pf, w=w, ix=0, iy=1, iz=0)

    assert isinstance(layer, ParticleEnsemble)
    assert layer.bottom == grid.levels[0]
    assert layer.top == grid.levels[1]
    assert layer.w_ref == w

    tau_ref = layer.tau_ref.m_as("dimensionless")
    assert tau_ref.shape == (grid.n_cells_x, grid.n_cells_y)

    mask = np.zeros_like(tau_ref, dtype=bool)
    mask[0, 1] = True
    layer_height_km = np.diff(grid.levels.m_as("km"))[0]
    expected_tau = (
        2.0 * 0.5 * layer_height_km
    )  # (ext_value * mass_concentration) * height
    np.testing.assert_allclose(tau_ref[mask], expected_tau)
    np.testing.assert_allclose(tau_ref[~mask], 0.0)


def test_to_particle_ensemble_fixed_composition_matches_field(mode_mono):
    """
    When the fixed (reff, veff) matches the field's own (spatially uniform)
    composition exactly, per-voxel sigma_t must match the field's own
    eval_sigma_t exactly: the distribution stores raw per-voxel tau on the
    field's own grid, so eval_fractions' 'nearest' lookup hits the query
    points exactly.
    """
    pf = make_particle_field(
        n_x=2,
        n_y=2,
        n_z=3,
        w=(1.0,),
        reff=10.0,
        veff=0.1,
        mass_concentration=0.5,
        ext_value=2.0,
        ssa_value=0.6,
    )
    grid = pf.geometry.grid
    w = ureg.Quantity(1.0, "micron")
    si = MonoSpectralIndex(w=w)

    ensemble = ParticleField.to_particle_ensemble_fixed_composition(
        pf,
        w=w,
        i_reff=0,
        i_veff=0,
    )

    assert isinstance(ensemble, ParticleEnsemble)
    assert ensemble.tau_ref.m_as("dimensionless").shape == (
        grid.n_cells_x,
        grid.n_cells_y,
    )
    assert ensemble.bottom == grid.levels[0]
    assert ensemble.top == grid.levels[-1]

    sigma_t_field = pf.eval_sigma_t(si, grid).m_as("1/km")
    sigma_t_ensemble = ensemble.eval_sigma_t(si, grid).m_as("1/km")
    np.testing.assert_allclose(sigma_t_ensemble, sigma_t_field, rtol=1e-8)


def test_to_particle_ensemble_fixed_composition_irregular_z(mode_mono):
    """Density is preserved exactly when the source profile has an irregular z-grid."""
    z_levels_src = [0.0, 1.0, 1.5, 4.0]
    target_levels = np.linspace(0.0, 4.0, 9)
    pf, grid = _make_z_profile_particle_field(z_levels_src, target_levels=target_levels)
    w = ureg.Quantity(1.0, "micron")
    si = MonoSpectralIndex(w=w)

    ensemble = ParticleField.to_particle_ensemble_fixed_composition(
        pf,
        w=w,
        i_reff=0,
        i_veff=0,
    )

    sigma_t_field = pf.eval_sigma_t(si, grid).m_as("1/km")
    sigma_t_ensemble = ensemble.eval_sigma_t(si, grid).m_as("1/km")
    np.testing.assert_allclose(sigma_t_ensemble, sigma_t_field, rtol=1e-8)


def test_to_particle_ensemble_fixed_composition_zero_density_column(mode_mono):
    """A zero mass_concentration field yields a zero tau_ref ensemble."""
    pf = make_particle_field(n_x=2, n_y=1, n_z=2, w=(1.0,), mass_concentration=0.0)
    w = ureg.Quantity(1.0, "micron")

    ensemble = ParticleField.to_particle_ensemble_fixed_composition(
        pf,
        w=w,
        i_reff=0,
        i_veff=0,
    )

    np.testing.assert_allclose(ensemble.tau_ref.m_as("dimensionless"), 0.0)


def test_particle_field_properties_requires_size_distribution():
    """Constructing a ParticleField with a properties dataset lacking reff/veff raises."""
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0]) * ureg.km,
    ).centered()
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    profile = make_profile_dataset(grid)

    mu = np.linspace(-1.0, 1.0, 5)
    theta = np.rad2deg(np.arccos(mu))
    properties_no_size_dist = make_aer_core_v2(
        w=ureg.Quantity([1.0], "micron"),
        phamat=["11"],
        mu=ureg.Quantity(mu[np.newaxis, :], "dimensionless"),
        theta=ureg.Quantity(theta[np.newaxis, :], "degree"),
        ext=ureg.Quantity([1.0], "1/km"),
        ssa=ureg.Quantity([0.6], "dimensionless"),
        phase=ureg.Quantity(np.ones((1, 1, 5)), "1/sr"),
        pmom=np.zeros((1, 1)),
    )

    with pytest.raises(ValueError, match="reff.*veff"):
        ParticleField(
            profile=profile, properties=properties_no_size_dist, geometry=geometry
        )


def test_from_particle_ensemble_roundtrip(mode_mono):
    """from_particle_ensemble preserves total sigma_t and albedo relative to the source ensemble."""
    layer = make_particle_ensemble(sigma_t_val=1.0, ssa_val=0.6)
    si_ref = MonoSpectralIndex(w=layer.w_ref)

    field = ParticleField.from_particle_ensemble(
        layer,
        mass_concentration=ureg.Quantity(1e-3, "g/m^3"),
        reff=ureg.Quantity(10.0, "micron"),
        veff=ureg.Quantity(0.1, "dimensionless"),
    )

    assert isinstance(field, ParticleField)
    np.testing.assert_allclose(field.properties.reff.m_as("micron"), [10.0])
    np.testing.assert_allclose(field.properties.veff.m_as(ureg.dimensionless), [0.1])

    layer_sigma_t_sum = layer.eval_sigma_t(si_ref).m_as("1/km").sum()
    field_sigma_t = field.eval_sigma_t(si_ref)
    np.testing.assert_allclose(
        field_sigma_t.m_as("1/km").sum(), layer_sigma_t_sum, rtol=1e-6
    )

    field_albedo = field.eval_albedo(si_ref)
    mask = field_sigma_t.m > 0
    assert np.any(mask)
    np.testing.assert_allclose(field_albedo.m_as("dimensionless")[mask], 0.6, rtol=1e-6)


def test_from_particle_ensemble_custom_grid(mode_mono):
    """from_particle_ensemble evaluates onto an explicitly given grid instead of the ensemble's own."""
    layer = make_particle_ensemble()
    custom_grid = PlaneParallelGridCoords(
        edges_x=np.array([-2.0, 2.0]) * ureg.km,
        edges_y=np.array([-2.0, 2.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0]) * ureg.km,
    )

    field = ParticleField.from_particle_ensemble(
        layer,
        mass_concentration=ureg.Quantity(1e-3, "g/m^3"),
        reff=ureg.Quantity(10.0, "micron"),
        veff=ureg.Quantity(0.1, "dimensionless"),
        grid=custom_grid,
    )

    np.testing.assert_allclose(
        field.geometry.grid.edges_x.m_as("km"), custom_grid.edges_x.m_as("km")
    )


def test_theta_convention_consistency(mode_mono):
    """to_particle_ensemble/from_particle_ensemble preserve theta/mu ordering through a round trip."""
    theta_values_desc = np.array([180.0, 150.0, 100.0, 20.0, 0.0])
    mu_values_asc = np.cos(np.deg2rad(theta_values_desc))
    n_theta = len(theta_values_desc)

    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0]) * ureg.km,
    )
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])
    profile = make_profile_dataset(grid, reff=10.0, veff=0.1, mass_concentration=1e-3)

    properties = make_aer_core_v2(
        w=np.array([1.0]) * ureg.micron,
        phamat=PHAMAT_LABELS[4],
        mu=ureg.Quantity(
            mu_values_asc[np.newaxis, np.newaxis, np.newaxis, :], "dimensionless"
        ),
        theta=ureg.Quantity(
            theta_values_desc[np.newaxis, np.newaxis, np.newaxis, :], "degree"
        ),
        ext=np.full((1, 1, 1), 1.0) * ureg("km^-1 / (g / m^3)"),
        ssa=np.full((1, 1, 1), 0.8) * ureg.dimensionless,
        phase=ureg.Quantity(
            np.broadcast_to(theta_values_desc, (4, 1, 1, 1, n_theta)).copy(), "1/sr"
        ),
        # Bypass automatic pmom computation: the raw (unnormalized) phase
        # values are a deliberate proxy for round-trip checks below, not a
        # physically valid phase function.
        pmom=np.zeros((1, 1, 1, 1)),
        reff=np.array([10.0]) * ureg.micron,
        veff=np.array([0.1]) * ureg.dimensionless,
    )
    pf = ParticleField(profile=profile, properties=properties, geometry=geometry)

    layer = ParticleField.to_particle_ensemble(
        pf, w=ureg.Quantity(1.0, "micron"), ix=0, iy=0, iz=0
    )

    pp = layer.particle_properties
    mu = pp.data["mu"].values[0]
    assert np.all(np.diff(mu) > 0)
    np.testing.assert_allclose(pp.phase.values[0, :, 0], np.rad2deg(np.arccos(mu)))

    field2 = ParticleField.from_particle_ensemble(
        layer,
        mass_concentration=ureg.Quantity(1e-3, "g/m^3"),
        reff=ureg.Quantity(10.0, "micron"),
        veff=ureg.Quantity(0.1, "dimensionless"),
    )

    np.testing.assert_allclose(
        field2.properties.data["theta"].values.ravel(), theta_values_desc
    )
    np.testing.assert_allclose(
        field2.properties.data["phase"].values[0].ravel(), theta_values_desc
    )


def test_multi_reff_veff_ws(mode_mono):
    """End-to-end check with genuinely multi-reff, multi-veff, and multi-wavelength data."""
    reff_vals = np.array([10.0, 20.0])
    veff_vals = np.array([0.1, 0.2])
    w_nodes = np.array([0.5, 1.0, 1.5])
    n_theta = 5
    nreff, nveff, nw = 2, 2, 3

    mu_1d = np.linspace(-1.0, 1.0, n_theta)
    mu = np.tile(mu_1d, (nw, nreff, nveff, 1))
    theta = np.degrees(np.arccos(mu))
    phase = np.ones((4, nw, nreff, nveff, n_theta))

    ext = np.zeros((nw, nreff, nveff))
    ssa = np.zeros((nw, nreff, nveff))
    ext[:, 0, 0], ext[:, 1, 0] = 2.0, 4.0
    ext[:, 0, 1], ext[:, 1, 1] = 3.0, 6.0
    ssa[:, :, 0] = 0.6
    ssa[:, :, 1] = 0.9

    properties = make_aer_core_v2(
        w=w_nodes * ureg.micron,
        phamat=PHAMAT_LABELS[4],
        mu=mu * ureg.dimensionless,
        theta=theta * ureg.degree,
        ext=ext * ureg("km^-1 / (g / m^3)"),
        ssa=ssa * ureg.dimensionless,
        phase=phase * ureg("1/sr"),
        reff=reff_vals * ureg.micron,
        veff=veff_vals * ureg.dimensionless,
    )
    assert properties.sizes["reff"] == 2
    assert properties.sizes["veff"] == 2
    assert properties.sizes["w"] == 3

    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0]) * ureg.km,
    ).centered()
    geometry = PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])

    profile = make_profile_dataset(grid, reff=10.0, veff=0.15, mass_concentration=0.5)

    pf = ParticleField(profile=profile, properties=properties, geometry=geometry)

    si = MonoSpectralIndex(w=ureg.Quantity(0.8, "micron"))
    shape = (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z)

    expected_ext = 0.5 * 2.0 + 0.5 * 3.0
    expected_ssa = 0.5 * 0.6 + 0.5 * 0.9

    sigma_t = pf.eval_sigma_t(si)
    assert sigma_t.shape == shape
    np.testing.assert_allclose(sigma_t.m_as("1/km"), expected_ext * 0.5, rtol=1e-6)

    albedo = pf.eval_albedo(si)
    assert albedo.shape == shape
    np.testing.assert_allclose(albedo.m_as("dimensionless"), expected_ssa, rtol=1e-6)

    phase = pf.phase
    assert isinstance(phase, ParticleFieldPhaseFunction)
    np.testing.assert_allclose(phase.r_eff_grid, reff_vals)
    np.testing.assert_allclose(phase.v_eff_grid, veff_vals)

    params = pf._params_phase
    for key in ["nodes", "phase_mueller", "grid_start", "grid_len", "sigma_s_weight"]:
        assert key in params
