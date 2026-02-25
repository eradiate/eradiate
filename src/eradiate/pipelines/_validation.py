"""Validation utilities for pipeline nodes.

This module provides reusable validators for common validation patterns,
especially for xarray DataArrays and numpy arrays.
"""

from typing import Optional

import numpy as np
import xarray as xr


def validate_type(expected_type):
    """Create a validator that checks output type.

    Parameters
    ----------
    expected_type : type
        Expected type of the value.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_type(np.ndarray)
    >>> validator(np.array([1, 2, 3]))  # OK
    >>> validator([1, 2, 3])  # Raises TypeError
    """

    def validator(value):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )

    return validator


def validate_dataarray_dims(required_dims):
    """Create a validator that checks DataArray dimensions.

    Parameters
    ----------
    required_dims : list of str
        Required dimension names.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_dataarray_dims(["x", "y"])
    >>> da = xr.DataArray(np.zeros((3, 4)), dims=["x", "y"])
    >>> validator(da)  # OK
    """

    def validator(value):
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"Expected xr.DataArray, got {type(value).__name__}")

        missing = set(required_dims) - set(value.dims)
        if missing:
            raise ValueError(f"DataArray missing dimensions: {missing}")

    return validator


def validate_dataarray_coords(required_coords):
    """Create a validator that checks DataArray coordinates.

    Parameters
    ----------
    required_coords : list of str
        Required coordinate names.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_dataarray_coords(["x", "y"])
    >>> da = xr.DataArray(np.zeros((3, 4)), coords={"x": [0, 1, 2], "y": [0, 1, 2, 3]})
    >>> validator(da)  # OK
    """

    def validator(value):
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"Expected xr.DataArray, got {type(value).__name__}")

        missing = set(required_coords) - set(value.coords)
        if missing:
            raise ValueError(f"DataArray missing coordinates: {missing}")

    return validator


def validate_shape(expected_shape):
    """Create a validator that checks array shape.

    Parameters
    ----------
    expected_shape : tuple
        Expected shape. Use None for wildcard dimensions.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_shape((3, None))  # First dim must be 3
    >>> validator(np.zeros((3, 4)))  # OK
    >>> validator(np.zeros((2, 4)))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "shape"):
            actual_shape = value.shape
            if len(expected_shape) != len(actual_shape):
                raise ValueError(
                    f"Shape mismatch: expected {len(expected_shape)} dimensions, "
                    f"got {len(actual_shape)}"
                )
            # Support wildcards (None) in expected_shape
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if expected is not None and expected != actual:
                    raise ValueError(
                        f"Shape mismatch at axis {i}: expected {expected}, got {actual}"
                    )

    return validator


def validate_range(min_val: Optional[float] = None, max_val: Optional[float] = None):
    """Create a validator that checks value range.

    Parameters
    ----------
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_range(min_val=0, max_val=1)
    >>> validator(np.array([0.5, 0.8]))  # OK
    >>> validator(np.array([-0.1, 0.5]))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.array(value)

        if min_val is not None and np.any(arr < min_val):
            raise ValueError(f"Values below minimum {min_val}")

        if max_val is not None and np.any(arr > max_val):
            raise ValueError(f"Values above maximum {max_val}")

    return validator


def validate_no_nan():
    """Create a validator that checks for NaN values.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_no_nan()
    >>> validator(np.array([1, 2, 3]))  # OK
    >>> validator(np.array([1, np.nan, 3]))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return

        if np.any(np.isnan(arr)):
            raise ValueError("Data contains NaN values")

    return validator


def validate_no_inf():
    """Create a validator that checks for infinite values.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_no_inf()
    >>> validator(np.array([1, 2, 3]))  # OK
    >>> validator(np.array([1, np.inf, 3]))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return

        if np.any(np.isinf(arr)):
            raise ValueError("Data contains infinite values")

    return validator


def validate_all_finite():
    """Create a validator that checks for both NaN and infinite values.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_all_finite()
    >>> validator(np.array([1, 2, 3]))  # OK
    >>> validator(np.array([1, np.nan, 3]))  # Raises ValueError
    >>> validator(np.array([1, np.inf, 3]))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return

        if not np.all(np.isfinite(arr)):
            raise ValueError("Data contains non-finite values (NaN or Inf)")

    return validator


def validate_positive():
    """Create a validator that checks all values are positive.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_positive()
    >>> validator(np.array([1, 2, 3]))  # OK
    >>> validator(np.array([0, 1, 2]))  # Raises ValueError (0 is not positive)
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.array(value)

        if not np.all(arr > 0):
            raise ValueError("All values must be positive (> 0)")

    return validator


def validate_non_negative():
    """Create a validator that checks all values are non-negative.

    Returns
    -------
    Callable
        Validator function.

    Examples
    --------
    >>> validator = validate_non_negative()
    >>> validator(np.array([0, 1, 2]))  # OK
    >>> validator(np.array([-1, 0, 1]))  # Raises ValueError
    """

    def validator(value):
        if hasattr(value, "values"):
            arr = value.values
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.array(value)

        if not np.all(arr >= 0):
            raise ValueError("All values must be non-negative (>= 0)")

    return validator
