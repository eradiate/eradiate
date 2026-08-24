import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.grid import PlaneParallelGridCoords, SphericalShellGridCoords
from eradiate.scenes.atmosphere import ParticleProfile


def make_profile_dataset(
    i_x, i_y, i_z, reff, veff, mass_concentration, x_levels, y_levels, z_levels
) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "i_x": (["index"], list(i_x)),
            "i_y": (["index"], list(i_y)),
            "i_z": (["index"], list(i_z)),
            "reff": (["index"], list(reff), {"units": "micron"}),
            "veff": (["index"], list(veff), {"units": "dimensionless"}),
            "mass_concentration": (
                ["index"],
                list(mass_concentration),
                {"units": "g/m^3"},
            ),
            "x_levels": (["x"], np.asarray(x_levels), {"units": "km"}),
            "y_levels": (["y"], np.asarray(y_levels), {"units": "km"}),
            "z_levels": (["z"], np.asarray(z_levels), {"units": "km"}),
        },
    )


class TestConstruction:
    def test_valid_dataset(self):
        """A dataset with all required variables constructs successfully."""
        ds = make_profile_dataset(
            i_x=[0],
            i_y=[0],
            i_z=[0],
            reff=[10.0],
            veff=[0.1],
            mass_concentration=[1e-3],
            x_levels=[0.0, 1.0],
            y_levels=[0.0, 1.0],
            z_levels=[0.0, 1.0],
        )
        ParticleProfile(data=ds)

    @pytest.mark.parametrize(
        "missing",
        [
            "i_x",
            "i_y",
            "i_z",
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
            i_x=[0],
            i_y=[0],
            i_z=[0],
            reff=[10.0],
            veff=[0.1],
            mass_concentration=[1e-3],
            x_levels=[0.0, 1.0],
            y_levels=[0.0, 1.0],
            z_levels=[0.0, 1.0],
        ).drop_vars(missing)
        with pytest.raises(ValueError, match="missing required variable"):
            ParticleProfile(data=ds)


class TestResampleRegular:
    @pytest.fixture
    def profile(self) -> ParticleProfile:
        return ParticleProfile(
            data=make_profile_dataset(
                i_x=[0, 1, 0, 1],
                i_y=[0, 0, 1, 1],
                i_z=[0, 0, 1, 1],
                reff=[10.0, 11.0, 12.0, 13.0],
                veff=[0.1, 0.1, 0.1, 0.1],
                mass_concentration=[1e-3, 2e-3, 3e-3, 4e-3],
                x_levels=[0.0, 1.0, 2.0],
                y_levels=[0.0, 1.0, 2.0],
                z_levels=[0.0, 1.0, 4.0],
            )
        )

    @pytest.fixture
    def grid(self) -> PlaneParallelGridCoords:
        return PlaneParallelGridCoords(
            edges_x=np.array([0.0, 1.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )

    def test_nearest_neighbour_z(self, profile, grid):
        """resample_regular picks each target z-layer's nearest source voxel."""
        result = profile.resample_regular(grid)

        points = {
            (int(ix), int(iy), int(iz)): float(r)
            for ix, iy, iz, r in zip(
                result.data["i_x"].values,
                result.data["i_y"].values,
                result.data["i_z"].values,
                result.data["reff"].values,
            )
        }

        np.testing.assert_allclose(points[(0, 0, 0)], 10.0)
        assert (0, 0, 1) not in points
        assert (0, 0, 2) not in points
        assert (0, 0, 3) not in points

        assert (0, 1, 0) not in points
        np.testing.assert_allclose(points[(0, 1, 1)], 12.0)
        np.testing.assert_allclose(points[(0, 1, 2)], 12.0)
        assert (0, 1, 3) not in points

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
        dim = axis[0]
        ds[axis] = ([dim], np.array([0.0, 1.0, 3.0]), {"units": "km"})
        irregular = ParticleProfile(data=ds)
        with pytest.raises(ValueError, match=f"'{axis}' must be regularly spaced"):
            irregular.resample_regular(grid)

    def test_extent_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose extent doesn't match the profile's."""
        mismatched_grid = PlaneParallelGridCoords(
            edges_x=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_resolution_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose resolution doesn't match the profile's."""
        mismatched_grid = PlaneParallelGridCoords(
            edges_x=np.array([0.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_spherical_shell_dispatch(self):
        """resample_regular dispatches to the SphericalShellGridCoords implementation."""
        profile = ParticleProfile(
            data=make_profile_dataset(
                i_x=[0],
                i_y=[0],
                i_z=[0],
                reff=[10.0],
                veff=[0.1],
                mass_concentration=[1e-3],
                x_levels=np.linspace(0.0, 2 * np.pi, 3),
                y_levels=np.linspace(0.0, np.pi, 3),
                z_levels=[0.0, 1.0],
            )
        )
        profile.data["x_levels"].attrs["units"] = "radian"
        profile.data["y_levels"].attrs["units"] = "radian"

        grid = SphericalShellGridCoords(
            levels=np.array([0.0, 1.0]) * ureg.km,
            azimuths=np.linspace(0.0, 2 * np.pi, 3) * ureg.radian,
            colatitudes=np.linspace(0.0, np.pi, 3) * ureg.radian,
        )
        result = profile.resample_regular(grid)
        assert result.data["i_x"].values.tolist() == [0]
