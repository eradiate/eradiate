"""Tests for validation utilities."""

import numpy as np
import pytest
import xarray as xr

from eradiate.pipelines import validation as pval


class TestValidateType:
    """Tests for validate_type."""

    def test_valid_type(self):
        """Test validation with correct type."""
        validator = pval.validate_type(int)
        validator(42)  # Should not raise

    def test_invalid_type_raises(self):
        """Test validation with incorrect type."""
        validator = pval.validate_type(int)
        with pytest.raises(TypeError, match="Expected int"):
            validator("not an int")

    def test_numpy_array(self):
        """Test validation with numpy array."""
        validator = pval.validate_type(np.ndarray)
        validator(np.array([1, 2, 3]))  # Should not raise

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_type(xr.DataArray)
        da = xr.DataArray([1, 2, 3])
        validator(da)  # Should not raise


class TestValidateDataArrayDims:
    """Tests for validate_dataarray_dims."""

    def test_valid_dims(self):
        """Test validation with correct dimensions."""
        validator = pval.validate_dataarray_dims(["x", "y"])
        da = xr.DataArray(np.zeros((3, 4)), dims=["x", "y"])
        validator(da)  # Should not raise

    def test_missing_dims_raises(self):
        """Test validation with missing dimensions."""
        validator = pval.validate_dataarray_dims(["x", "y", "z"])
        da = xr.DataArray(np.zeros((3, 4)), dims=["x", "y"])
        with pytest.raises(ValueError, match="missing dimensions"):
            validator(da)

    def test_extra_dims_allowed(self):
        """Test validation allows extra dimensions."""
        validator = pval.validate_dataarray_dims(["x"])
        da = xr.DataArray(np.zeros((3, 4)), dims=["x", "y"])
        validator(da)  # Should not raise

    def test_wrong_type_raises(self):
        """Test validation with wrong type."""
        validator = pval.validate_dataarray_dims(["x"])
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            validator(np.array([1, 2, 3]))


class TestValidateDataArrayCoords:
    """Tests for validate_dataarray_coords."""

    def test_valid_coords(self):
        """Test validation with correct coordinates."""
        validator = pval.validate_dataarray_coords(["x", "y"])
        da = xr.DataArray(
            np.zeros((3, 4)),
            coords={"x": [0, 1, 2], "y": [0, 1, 2, 3]},
        )
        validator(da)  # Should not raise

    def test_missing_coords_raises(self):
        """Test validation with missing coordinates."""
        validator = pval.validate_dataarray_coords(["x", "y", "z"])
        da = xr.DataArray(
            np.zeros((3, 4)),
            coords={"x": [0, 1, 2], "y": [0, 1, 2, 3]},
        )
        with pytest.raises(ValueError, match="missing coordinates"):
            validator(da)

    def test_wrong_type_raises(self):
        """Test validation with wrong type."""
        validator = pval.validate_dataarray_coords(["x"])
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            validator(np.array([1, 2, 3]))


class TestValidateShape:
    """Tests for validate_shape."""

    def test_valid_shape(self):
        """Test validation with correct shape."""
        validator = pval.validate_shape((3, 4))
        validator(np.zeros((3, 4)))  # Should not raise

    def test_invalid_shape_raises(self):
        """Test validation with incorrect shape."""
        validator = pval.validate_shape((3, 4))
        with pytest.raises(ValueError, match="Shape mismatch"):
            validator(np.zeros((3, 5)))

    def test_wildcard_dimension(self):
        """Test validation with wildcard dimension."""
        validator = pval.validate_shape((3, None))
        validator(np.zeros((3, 4)))  # Should not raise
        validator(np.zeros((3, 100)))  # Should not raise

    def test_wrong_ndim_raises(self):
        """Test validation with wrong number of dimensions."""
        validator = pval.validate_shape((3, 4))
        with pytest.raises(ValueError, match="expected 2 dimensions"):
            validator(np.zeros((3, 4, 5)))

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_shape((3, 4))
        da = xr.DataArray(np.zeros((3, 4)))
        validator(da)  # Should not raise


class TestValidateRange:
    """Tests for validate_range."""

    def test_valid_range(self):
        """Test validation with values in range."""
        validator = pval.validate_range(min_val=0, max_val=10)
        validator(np.array([0, 5, 10]))  # Should not raise

    def test_below_min_raises(self):
        """Test validation with values below minimum."""
        validator = pval.validate_range(min_val=0)
        with pytest.raises(ValueError, match="below minimum"):
            validator(np.array([-1, 0, 1]))

    def test_above_max_raises(self):
        """Test validation with values above maximum."""
        validator = pval.validate_range(max_val=10)
        with pytest.raises(ValueError, match="above maximum"):
            validator(np.array([9, 10, 11]))

    def test_only_min(self):
        """Test validation with only minimum."""
        validator = pval.validate_range(min_val=0)
        validator(np.array([0, 5, 100]))  # Should not raise

    def test_only_max(self):
        """Test validation with only maximum."""
        validator = pval.validate_range(max_val=10)
        validator(np.array([-100, 5, 10]))  # Should not raise

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_range(min_val=0, max_val=1)
        da = xr.DataArray([0.0, 0.5, 1.0])
        validator(da)  # Should not raise


class TestValidateNoNaN:
    """Tests for validate_no_nan."""

    def test_no_nan(self):
        """Test validation with no NaN values."""
        validator = pval.validate_no_nan()
        validator(np.array([1, 2, 3]))  # Should not raise

    def test_with_nan_raises(self):
        """Test validation with NaN values."""
        validator = pval.validate_no_nan()
        with pytest.raises(ValueError, match="NaN"):
            validator(np.array([1, np.nan, 3]))

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_no_nan()
        da = xr.DataArray([1, 2, 3])
        validator(da)  # Should not raise

    def test_xarray_with_nan_raises(self):
        """Test validation with xarray DataArray containing NaN."""
        validator = pval.validate_no_nan()
        da = xr.DataArray([1, np.nan, 3])
        with pytest.raises(ValueError, match="NaN"):
            validator(da)


class TestValidateNoInf:
    """Tests for validate_no_inf."""

    def test_no_inf(self):
        """Test validation with no infinite values."""
        validator = pval.validate_no_inf()
        validator(np.array([1, 2, 3]))  # Should not raise

    def test_with_inf_raises(self):
        """Test validation with infinite values."""
        validator = pval.validate_no_inf()
        with pytest.raises(ValueError, match="infinite"):
            validator(np.array([1, np.inf, 3]))

    def test_with_neg_inf_raises(self):
        """Test validation with negative infinite values."""
        validator = pval.validate_no_inf()
        with pytest.raises(ValueError, match="infinite"):
            validator(np.array([1, -np.inf, 3]))


class TestValidateAllFinite:
    """Tests for validate_all_finite."""

    def test_all_finite(self):
        """Test validation with all finite values."""
        validator = pval.validate_all_finite()
        validator(np.array([1, 2, 3]))  # Should not raise

    def test_with_nan_raises(self):
        """Test validation with NaN values."""
        validator = pval.validate_all_finite()
        with pytest.raises(ValueError, match="non-finite"):
            validator(np.array([1, np.nan, 3]))

    def test_with_inf_raises(self):
        """Test validation with infinite values."""
        validator = pval.validate_all_finite()
        with pytest.raises(ValueError, match="non-finite"):
            validator(np.array([1, np.inf, 3]))

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_all_finite()
        da = xr.DataArray([1.0, 2.0, 3.0])
        validator(da)  # Should not raise


class TestValidatePositive:
    """Tests for validate_positive."""

    def test_all_positive(self):
        """Test validation with all positive values."""
        validator = pval.validate_positive()
        validator(np.array([1, 2, 3]))  # Should not raise

    def test_with_zero_raises(self):
        """Test validation with zero."""
        validator = pval.validate_positive()
        with pytest.raises(ValueError, match="must be positive"):
            validator(np.array([0, 1, 2]))

    def test_with_negative_raises(self):
        """Test validation with negative values."""
        validator = pval.validate_positive()
        with pytest.raises(ValueError, match="must be positive"):
            validator(np.array([-1, 0, 1]))

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_positive()
        da = xr.DataArray([1, 2, 3])
        validator(da)  # Should not raise


class TestValidateNonNegative:
    """Tests for validate_non_negative."""

    def test_all_non_negative(self):
        """Test validation with all non-negative values."""
        validator = pval.validate_non_negative()
        validator(np.array([0, 1, 2]))  # Should not raise

    def test_with_negative_raises(self):
        """Test validation with negative values."""
        validator = pval.validate_non_negative()
        with pytest.raises(ValueError, match="non-negative"):
            validator(np.array([-1, 0, 1]))

    def test_xarray_dataarray(self):
        """Test validation with xarray DataArray."""
        validator = pval.validate_non_negative()
        da = xr.DataArray([0, 1, 2])
        validator(da)  # Should not raise
