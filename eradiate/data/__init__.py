"""Data handling facilities.

A typical data handling pattern uses the :func:`open` function.
This function can be called either through its first two parameters
``category`` and ``id``, or through its third parameter ``path``. The first
kind of call will search Eradiate's data registry for a valid data set;
the second kind of call will try and resolve directly a path using the
:class:`.PathResolver`.

.. admonition:: Example: first-kind call to ``open()``

   The following code accesses the Thuillier irradiance spectrum
   :cite:`Thuillier2003SolarSpectralIrradiance`:

   .. code:: python

      import eradiate.data as data

      ds = data.open("solar_irradiance_spectrum", "thuillier_2003")


.. admonition:: Example: second-kind call to ``open()``

   The following code accesses the Thuillier irradiance spectrum
   :cite:`Thuillier2003SolarSpectralIrradiance`:

   .. code:: python

      import eradiate.data as data

      ds = data.open(path="spectra/thuillier_2003.nc")
"""

import os

import xarray as xr

from .absorption_spectra import _AbsorptionGetter
from .solar_irradiance_spectra import _SolarIrradianceGetter
from ..util.presolver import PathResolver

_presolver = PathResolver()

_getters = {
    "absorption_spectrum": _AbsorptionGetter,
    "solar_irradiance_spectrum": _SolarIrradianceGetter
}


def open(category=None, id=None, path=None):
    """Opens a data set.

    Parameter ``category`` (str or None):
        If ``None``, ``path`` must not be ``None`` .
        Dataset category identifier. Valid data set categories are:

        * :class:`solar_irradiance_spectrum <eradiate.data.solar_irradiance_spectra>`
        * :class:`absorption_cross_section_spectrum <eradiate.data.absorption_spectra>`

    Parameter ``id`` (str or None):
        If ``None``, ``path`` must not be ``None`` .
        Dataset identifier inside a given category. See category documentation
        for valid ID values.

    Parameter ``path`` (path-like or None):
        If not ``None``, takes precedence over ``category`` and ``id``.
        Path to the requested resource, resolved by the :class:`.PathResolver`.

    Returns → :class:`xarray.Dataset`:
        Dataset.

    Raises → ValueError:
        The requested resource is not handled.
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
            return getter.open(id)
        except ValueError:
            raise

    # path is not None: we open the data
    fname = _presolver.resolve(path)
    ext = os.path.splitext(fname)[1]

    if ext == ".nc":
        return xr.open_dataset(fname)

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
