"""Data handling facilities.

A typical data loading pattern uses the :func:`load` function.
This function can be called either through its first two parameters
``category`` and ``id``, or through its third parameter ``path``. The first
kind of call will search Eradiate's data registry for a valid data set;
the second kind of call will try and resolve directly a path using the
:class:`.PathResolver`.

.. admonition:: Example: first-kind call to ``load()``

   The following code accesses the Thuillier irradiance spectrum
   :cite:`Thuillier2003SolarSpectralIrradiance`:

   .. code:: python

      import eradiate.data as data

      ds = data.load("solar_irradiance_spectrum", "thuillier_2003")


.. admonition:: Example: second-kind call to ``load()``

   The following code accesses the Thuillier irradiance spectrum
   :cite:`Thuillier2003SolarSpectralIrradiance`:

   .. code:: python

      import eradiate.data as data

      ds = data.load(path="spectra/thuillier_2003.nc")
"""

import functools
import os

import xarray as xr

from .solar_irradiance_spectra import _SolarIrradianceGetter
from ..util.presolver import PathResolver

_presolver = PathResolver()

_getters = {
    "solar_irradiance_spectrum": _SolarIrradianceGetter
}


@functools.lru_cache(maxsize=32)
def load(category=None, id=None, path=None):
    """Load a data set. Results produced by this
    function are cached using an
    `LRU <https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)>`_
    policy to minimise hard drive access.

    Parameter ``category`` (str or None):
        If ``None``, ``path`` must not be ``None`` .
        Dataset category identifier. Valid data set categories are:

        * :class:`solar_irradiance_spectrum <eradiate.data.solar_irradiance_spectra>`

    Parameter ``id`` (str or None):
        If ``None``, ``path`` must not be ``None`` .
        Dataset identifier inside a given category. See category documentation
        for valid ID values.

    Parameter ``path`` (path-like or None):
        If not ``None``, takes precedence over ``category`` and ``id``.
        Path to the requested resource, resolved by the :class:`.PathResolver`.

    Returns → :class:`xarray.Dataset`:
        Loaded dataset.

    Raises → ValueError:
        The requested resource is not handled by this loader.
    """
    if path is None:
        if category is None or id is None:
            raise ValueError("if 'path' is None, 'category' and 'id' must not "
                             "be None")

        try:
            getter = _getters[category]
        except KeyError:
            raise ValueError(f"invalid data category '{category}'")

        try:
            path = getter.path(id)
        except ValueError:
            raise

    fname = _presolver.resolve(path)
    ext = os.path.splitext(fname)[1]

    if ext == ".nc":
        return xr.load_dataset(fname)

    raise ValueError(f"cannot load resource {fname}")


def registered(category):
    """Get a list of registered dataset IDs for a given data set category.

    Parameter ``category`` (str):
        Dataset category identifier. See :func:`load` for valid categories.

    Returns → list[str]:
        List of registered data set IDs for the selected category.

    Raises → ValueError:
        Unknown requested category.
    """
    try:
        getter = _getters[category]
    except KeyError:
        raise ValueError(f"invalid data category '{category}'")

    return getter.registered()

# TODO: add functions to check if data is missing
