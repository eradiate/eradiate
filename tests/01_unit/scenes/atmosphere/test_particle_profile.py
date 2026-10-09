import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid, SphericalGrid
from eradiate.scenes.atmosphere import ParticleProfile


def make_profile_dataset(
    reff, veff, mass_concentration, x_levels, y_levels, z_levels
) -> xr.Dataset:
    """
    Build a dense Pfd-Core v1 profile dataset. ``reff``/``veff``/
    ``mass_concentration`` must be broadcastable to shape
    ``(len(z_levels) - 1, len(y_levels) - 1, len(x_levels) - 1)``.
    """
    shape = (len(z_levels) - 1, len(y_levels) - 1, len(x_levels) - 1)
    return xr.Dataset(
        data_vars={
            "reff": (
                ["z", "y", "x"],
                np.broadcast_to(reff, shape).copy(),
                {"units": "micron"},
            ),
            "veff": (
                ["z", "y", "x"],
                np.broadcast_to(veff, shape).copy(),
                {"units": "dimensionless"},
            ),
            "mass_concentration": (
                ["z", "y", "x"],
                np.broadcast_to(mass_concentration, shape).copy(),
                {"units": "g/m^3"},
            ),
            "x_levels": (["x_levels"], np.asarray(x_levels), {"units": "km"}),
            "y_levels": (["y_levels"], np.asarray(y_levels), {"units": "km"}),
            "z_levels": (["z_levels"], np.asarray(z_levels), {"units": "km"}),
        },
    )


class TestConstruction:
    def test_valid_dataset(self):
        """A dataset with all required variables constructs successfully."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_levels=[0.0, 1.0],
            y_levels=[0.0, 1.0],
            z_levels=[0.0, 1.0],
        )
        ParticleProfile(data=ds)

    @pytest.mark.parametrize(
        "missing",
        [
            "reff",
            "veff",
            "mass_concentration",
            "x_levels",
            "y_levels",
            "z_levels",
        ],
    )
    def test_missing_variable_raises(self, missing):
        """Constructing from a dataset missing any required variable raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_levels=[0.0, 1.0],
            y_levels=[0.0, 1.0],
            z_levels=[0.0, 1.0],
        ).drop_vars(missing)
        with pytest.raises(ValueError, match="missing required variable"):
            ParticleProfile(data=ds)


class TestVariablesAndGetitem:
    @pytest.fixture
    def profile(self) -> ParticleProfile:
        return ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_levels=[0.0, 1.0],
                y_levels=[0.0, 1.0],
                z_levels=[0.0, 1.0],
            )
        )

    def test_variables_excludes_structural(self, profile):
        """variables lists non-structural data_vars only."""
        assert set(profile.variables) == {"reff", "veff", "mass_concentration"}

    def test_getitem_matches_named_property(self, profile):
        """profile[name] returns the same cached value as the matching property."""
        assert profile["reff"] is profile.reff
        assert profile["veff"] is profile.veff
        assert profile["mass_concentration"] is profile.mass_concentration

    def test_getitem_rejects_structural_variable(self, profile):
        """profile[name] raises for a structural (x/y/z_levels) variable."""
        with pytest.raises(KeyError, match="x_levels"):
            profile["x_levels"]

    def test_getitem_rejects_unknown_variable(self, profile):
        """profile[name] raises for a name that isn't a profile variable at all."""
        with pytest.raises(KeyError, match="nonexistent"):
            profile["nonexistent"]


class TestResampleRegular:
    @pytest.fixture
    def profile(self) -> ParticleProfile:
        reff = [[[10.0, 11.0], [0.0, 0.0]], [[0.0, 0.0], [12.0, 13.0]]]
        mass_concentration = [
            [[1e-3, 2e-3], [0.0, 0.0]],
            [[0.0, 0.0], [3e-3, 4e-3]],
        ]
        return ParticleProfile(
            data=make_profile_dataset(
                reff=reff,
                veff=0.1,
                mass_concentration=mass_concentration,
                x_levels=[0.0, 1.0, 2.0],
                y_levels=[0.0, 1.0, 2.0],
                z_levels=[0.0, 1.0, 4.0],
            )
        )

    @pytest.fixture
    def grid(self) -> CartesianGrid:
        return CartesianGrid(
            edges_x=np.array([0.0, 1.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )

    def test_nearest_neighbour_z(self, profile, grid):
        """resample_regular picks each target z-layer's nearest source voxel."""
        result = profile.resample_regular(grid)
        reff = result.data["reff"].values
        mass_concentration = result.data["mass_concentration"].values

        np.testing.assert_allclose(reff[0, 0, 0], 10.0)
        assert mass_concentration[0, 0, 0] > 0
        assert mass_concentration[1, 0, 0] == 0
        assert mass_concentration[2, 0, 0] == 0
        assert mass_concentration[3, 0, 0] == 0

        assert mass_concentration[0, 1, 0] == 0
        np.testing.assert_allclose(reff[1, 1, 0], 12.0)
        assert mass_concentration[1, 1, 0] > 0
        np.testing.assert_allclose(reff[2, 1, 0], 12.0)
        assert mass_concentration[2, 1, 0] > 0
        assert mass_concentration[3, 1, 0] == 0

    def test_output_is_regular_on_all_axes(self, profile, grid):
        """The resampled profile's x/y/z levels are all evenly spaced."""
        result = profile.resample_regular(grid)
        for name in ("x_levels", "y_levels", "z_levels"):
            values = result.data[name].values
            step = np.diff(values)
            np.testing.assert_allclose(step, step[0])

    def test_output_dataset_valid(self, profile, grid):
        """The resampled dataset is itself a valid ParticleProfile input."""
        result = profile.resample_regular(grid)
        ParticleProfile(data=result.data)

    @pytest.mark.parametrize("axis", ["x_levels", "y_levels"])
    def test_irregular_source_raises(self, profile, grid, axis):
        """resample_regular rejects a profile with an irregular x_levels/y_levels axis."""
        ds = profile.data.copy()
        ds[axis] = ([axis], np.array([0.0, 1.0, 3.0]), {"units": "km"})
        irregular = ParticleProfile(data=ds)
        with pytest.raises(ValueError, match=f"'{axis}' must be regularly spaced"):
            irregular.resample_regular(grid)

    def test_extent_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose extent doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            edges_x=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_resolution_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose resolution doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            edges_x=np.array([0.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_spherical_shell_dispatch(self):
        """resample_regular dispatches to the SphericalGrid implementation."""
        profile = ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_levels=np.linspace(0.0, 2 * np.pi, 3),
                y_levels=np.linspace(0.0, np.pi, 3),
                z_levels=[0.0, 1.0],
            )
        )
        profile.data["x_levels"].attrs["units"] = "radian"
        profile.data["y_levels"].attrs["units"] = "radian"

        grid = SphericalGrid(
            levels=np.array([0.0, 1.0]) * ureg.km,
            azimuths=np.linspace(0.0, 2 * np.pi, 3) * ureg.radian,
            colatitudes=np.linspace(0.0, np.pi, 3) * ureg.radian,
        )
        result = profile.resample_regular(grid)
        assert result.data["reff"].shape == (1, 2, 2)


class TestToReffVeffVolumes:
    def test_empty_voxels_are_nan(self):
        """Voxels with zero mass_concentration carry NaN, regardless of their stored reff/veff."""
        ds = make_profile_dataset(
            reff=[[[10.0, 10.0]]],
            veff=0.1,
            mass_concentration=[[[1e-3, 0.0]]],
            x_levels=[0.0, 1.0, 2.0],
            y_levels=[0.0, 1.0],
            z_levels=[0.0, 1.0],
        )
        profile = ParticleProfile(data=ds)
        reff_volume, veff_volume = profile.to_reff_veff_volumes()

        assert reff_volume.shape == (2, 1, 1)
        np.testing.assert_allclose(reff_volume.m_as("micron")[0, 0, 0], 10.0)
        assert np.isnan(reff_volume.m_as("micron")[1, 0, 0])
        assert np.isnan(veff_volume.m_as(ureg.dimensionless)[1, 0, 0])
