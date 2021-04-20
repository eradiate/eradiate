import numpy as np
import xarray as xr

from ... import data
from ...frame import direction_to_angles
from ...scenes.measure._distant import DistantMeasure
from ...warp import square_to_uniform_hemisphere


def apply_srf(da, srf):
    r"""Computes the weighted average of some data with a wavelength dimension,
    where the weights are taken from the values of an instrument spectral
    response function.

    If the values to average are denoted :math:`x` and the weights :math:`w`,
    then the returned value is:

    .. math::
        \frac{
            \sum_{i=0}^{n-1} w_i x_i
        }{
            \sum_{i=0}^{n-1} w_i
        }

    where :math:`n` is the number of wavelength values.


    .. admonition:: Example of use
        :class: hint

        Say you ran a simulation that computed some radiance values and returned
        these values in a ``results`` data arrray. In the simulation, the
        sensor was assumed to have a uniform spectral response function.
        In order to simulate the observations from a particular instrument
        onboard a satellite, e.g. the SLSTR instrument onboard Sentinel-3A,
        in the channel 1 (515 - 595 nm), you need to take into account the
        spectral response function of the instrument.
        To apply the spectral response function on ``results``, simply call:

        .. code:: python

            apply_srf(results, "sentinel_3a-slstr-1")


    Parameter ``da`` (:class:`~xarray.DataArray`):
        A data array with (at least) a wavelength dimension (labeled ``"w"``).

    Parameter ``srf`` (str):
        Spectral response function datasets identifier.

        See the list of available spectral response function datasets in the
        :class:`~eradiate.data.spectral_response_function`
        module.

    Return → :class:`~xarray.DataArray`:
        Weighted-average of the data along the wavelength dimension.
    """
    dataset = data.open(category="spectral_response_function", id=srf)
    weights = dataset.srf.interp(w=da.w.values)
    weighted = da.weighted(weights)
    return weighted.sum(dim="w")
