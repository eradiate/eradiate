import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid
from eradiate.scenes.atmosphere import ParticleExtinctionComponent
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import IsotropicPhaseFunction
from eradiate.spectral.index import SpectralIndex
from eradiate.testing.types import check_scene_element
from eradiate.units import symbol

# ------------------------------------------------------------------------------
#                                   Fixtures
# ------------------------------------------------------------------------------


def _make_grid():
    return CartesianGrid(
        x_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
        y_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
        levels=np.linspace(0.0, 4.0, 5) * ureg.km,
    )


def _make_geometry(grid):
    return PlaneParallelGeometry(toa_altitude=grid.levels[-1], grid=grid)


def _make_data(grid, w, sigma_t, albedo):
    """
    Build (sigma_t, albedo) DataArrays with exactly (w, x, y, z) dims, the
    ``z`` coordinate set to ``grid``'s own layers, and values broadcast from
    ``sigma_t``/``albedo`` (scalars or arrays compatible with
    ``(len(w), *grid.shape)``).
    """
    nx, ny, nz = grid.shape
    shape = (len(w), nx, ny, nz)

    def _da(value, units):
        data = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
        da = xr.DataArray(
            data,
            dims=("w", "x", "y", "z"),
            coords={"w": w.m, "z": grid.layers.m_as("km")},
            attrs={"units": units},
        )
        da["w"].attrs["units"] = str(w.units)
        da["z"].attrs["units"] = "km"
        return da

    return _da(sigma_t, "km^-1"), _da(albedo, "")


# ------------------------------------------------------------------------------
#                                   Tests
# ------------------------------------------------------------------------------


class TestParticleExtinctionComponent:
    def test_basic(self, mode_mono):
        """Basic constructor pattern and kernel dictionary generation."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)
        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec={"type": "isotropic"},
        )
        check_scene_element(particles)

    def test_phase_spec_instance(self):
        """phase_spec accepts an already-constructed PhaseFunction instance."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)
        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec=IsotropicPhaseFunction(),
        )
        assert isinstance(particles.phase, IsotropicPhaseFunction)

    @pytest.mark.parametrize("drop_dim", ["w", "x", "y", "z"])
    def test_dims_validator_missing_dim(self, drop_dim):
        """Raises if sigma_t/albedo are missing one of (w, x, y, z)."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)
        sigma_t = sigma_t.isel({drop_dim: 0}, drop=True)

        with pytest.raises(ValueError, match="dimensions must be exactly"):
            ParticleExtinctionComponent(
                geometry=_make_geometry(grid),
                sigma_t=sigma_t,
                albedo=albedo,
                phase_spec={"type": "isotropic"},
            )

    def test_dims_validator_size_mismatch(self):
        """Raises if sigma_t and albedo have different sizes along a shared dim."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [500.0, 600.0] * ureg.nm, 1.0, 0.8)
        sigma_t = sigma_t.isel(w=[0])

        with pytest.raises(ValueError, match="must have matching sizes"):
            ParticleExtinctionComponent(
                geometry=_make_geometry(grid),
                sigma_t=sigma_t,
                albedo=albedo,
                phase_spec={"type": "isotropic"},
            )

    def test_albedo_bounds_validator(self):
        """Raises if albedo exceeds 1."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 1.5)

        with pytest.raises(ValueError, match=r"Albedo must be lower or equal to 1\.0"):
            ParticleExtinctionComponent(
                geometry=_make_geometry(grid),
                sigma_t=sigma_t,
                albedo=albedo,
                phase_spec={"type": "isotropic"},
            )

    def test_switches_both_disabled_raises(self):
        """Construction raises when both bypass switches are disabled."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)

        with pytest.raises(ValueError, match="at least one of"):
            ParticleExtinctionComponent(
                geometry=_make_geometry(grid),
                sigma_t=sigma_t,
                albedo=albedo,
                phase_spec={"type": "isotropic"},
                has_absorption=False,
                has_scattering=False,
            )

    @pytest.mark.parametrize(
        "has_absorption, has_scattering, expected_t, expected_a, expected_s",
        [
            (True, True, 1.0, 0.2, 0.8),
            (False, True, 0.8, 0.0, 0.8),
            (True, False, 0.2, 0.2, 0.0),
        ],
        ids=["both", "noabs", "noscat"],
    )
    def test_eval_switches(
        self,
        mode_mono,
        has_absorption,
        has_scattering,
        expected_t,
        expected_a,
        expected_s,
    ):
        """eval_sigma_t/_a/_s/_albedo respect has_absorption/has_scattering."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)
        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec={"type": "isotropic"},
            has_absorption=has_absorption,
            has_scattering=has_scattering,
        )

        si = SpectralIndex.new(w=550.0 * ureg.nm)
        np.testing.assert_allclose(
            particles.eval_sigma_t(si).m_as("1/km"), expected_t, equal_nan=False
        )
        np.testing.assert_allclose(
            particles.eval_sigma_a(si).m_as("1/km"), expected_a, equal_nan=False
        )
        np.testing.assert_allclose(
            particles.eval_sigma_s(si).m_as("1/km"), expected_s, equal_nan=False
        )

    def test_eval_radprops(self, mode_mono):
        """eval_radprops returns (x_cell, y_cell, z_layer) data with matching coords."""
        grid = _make_grid()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, 1.0, 0.8)
        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec={"type": "isotropic"},
        )

        ds = particles.eval_radprops(SpectralIndex.new(w=550.0 * ureg.nm))
        assert dict(ds.sizes) == {"x_cell": 2, "y_cell": 2, "z_layer": 4}
        for dim, cells in [
            ("x_cell", grid.x_centers),
            ("y_cell", grid.y_centers),
            ("z_layer", grid.layers),
        ]:
            assert ds[dim].attrs["units"] == symbol(cells.units)
            np.testing.assert_allclose(ds[dim].values, cells.magnitude)
        sigma_t_units = ureg(ds.sigma_t.attrs["units"])
        np.testing.assert_allclose(
            ds.sigma_t.sel(x_cell=0.5, y_cell=1.5) * sigma_t_units.m_as("1/km"), 1.0
        )

    def test_interp_wavelength(self, mode_mono):
        """sigma_t/albedo are linearly interpolated over w."""
        grid = _make_grid()
        w = np.array([500.0, 600.0]) * ureg.nm
        sigma_t_values = np.array([1.0, 2.0]).reshape(2, 1, 1, 1)
        sigma_t, albedo = _make_data(grid, w, sigma_t_values, 0.5)

        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec={"type": "isotropic"},
        )

        si = SpectralIndex.new(w=550.0 * ureg.nm)
        result = particles.eval_sigma_t(si).m_as("1/km")
        np.testing.assert_allclose(result, 1.5, equal_nan=False)

    def test_interp_altitude(self, mode_mono):
        """sigma_t is linearly interpolated in z against the grid's own layers."""
        grid = _make_grid()
        nx, ny, nz = grid.shape
        w = np.array([550.0]) * ureg.nm
        # sigma_t's own z-grid is coarser than, and distinct from, the
        # geometry's grid; sigma_t(z) = z (km) so interpolation is exact.
        z_src = np.array([0.0, 4.0])

        sigma_t = xr.DataArray(
            np.broadcast_to(z_src, (1, nx, ny, 2)).astype(float).copy(),
            dims=("w", "x", "y", "z"),
            coords={"w": w.m, "z": z_src},
            attrs={"units": "km^-1"},
        )
        sigma_t["w"].attrs["units"] = "nm"
        sigma_t["z"].attrs["units"] = "km"

        albedo = xr.DataArray(
            np.full((1, nx, ny, 2), 0.5),
            dims=("w", "x", "y", "z"),
            coords={"w": w.m, "z": z_src},
            attrs={"units": ""},
        )
        albedo["w"].attrs["units"] = "nm"
        albedo["z"].attrs["units"] = "km"

        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t,
            albedo=albedo,
            phase_spec={"type": "isotropic"},
        )

        si = SpectralIndex.new(w=550.0 * ureg.nm)
        result = particles.eval_sigma_t(si).m_as("1/km")
        expected = np.broadcast_to(grid.layers.m_as("km"), (nx, ny, nz))
        np.testing.assert_allclose(result, expected, equal_nan=False)

    @pytest.mark.parametrize(
        "dims", [("w", "x", "y", "z"), ("z", "y", "w", "x")], ids=["wxyz", "zywx"]
    )
    def test_xy_passthrough(self, mode_mono, dims):
        """x/y variation passes through unchanged, aligned with the grid, for any input dim order."""
        grid = _make_grid()
        nx, ny, nz = grid.shape
        xy_pattern = np.arange(nx * ny).reshape(nx, ny, 1).astype(float)
        sigma_t_values = np.broadcast_to(xy_pattern, (1, nx, ny, nz)).copy()
        sigma_t, albedo = _make_data(grid, [550.0] * ureg.nm, sigma_t_values, 0.5)

        particles = ParticleExtinctionComponent(
            geometry=_make_geometry(grid),
            sigma_t=sigma_t.transpose(*dims),
            albedo=albedo.transpose(*dims),
            phase_spec={"type": "isotropic"},
        )

        si = SpectralIndex.new(w=550.0 * ureg.nm)
        result = particles.eval_sigma_t(si).m_as("1/km")
        expected = np.broadcast_to(xy_pattern, (nx, ny, nz))
        np.testing.assert_allclose(result, expected, equal_nan=False)
