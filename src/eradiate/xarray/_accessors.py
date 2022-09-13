import numpy as np
import xarray as xr

from .interp import dataarray_to_rgb, film_to_angular


@xr.register_dataarray_accessor("ert")
class EradiateDataArrayAccessor:
    """
    Convenience wrapper for operations on :class:`~xarray.DataArray`
    instances. Accessed as a ``DataArray.ert`` property.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_angular(
        self,
        theta: np.typing.ArrayLike,
        phi: np.typing.ArrayLike,
        x_label: str = "x",
        y_label: str = "y",
        theta_label: str = "theta",
        phi_label: str = "phi",
    ) -> xr.DataArray:
        """
        Attempt interpolation of self to angular coordinates using
        :func:`~eradiate.xarray.interp.film_to_angular`.
        """
        return film_to_angular(
            self._obj,
            theta=theta,
            phi=phi,
            x_label=x_label,
            y_label=y_label,
            theta_label=theta_label,
            phi_label=phi_label,
        )

    def to_rgb(
        self,
        channels,
        normalize: bool = True,
        gamma_correction: bool = True,
    ) -> np.ndarray:
        """
        Generate a basic RGB image as a Numpy array from self using
        :func:`~eradiate.xarray.interp.dataarray_to_rgb`.
        """
        return dataarray_to_rgb(
            self._obj,
            channels,
            normalize=normalize,
            gamma_correction=gamma_correction,
        )
