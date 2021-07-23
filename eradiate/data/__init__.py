"""
Data handling facilities.

A typical data handling pattern uses the :func:`open` function.
This function can be called either through its first two parameters
``category`` and ``id``, or through its third parameter ``path``. The first
kind of call will search Eradiate's data registry for a valid data set;
the second kind of call will try and resolve directly a path using the
:class:`.PathResolver`.

.. admonition:: Examples

   The following code accesses the Thuillier irradiance spectrum
   :cite:`Thuillier2003SolarSpectralIrradiance`:

   .. code:: python

      import eradiate.data as data
      # first-kind call to ``open()``
      ds = data.open("solar_irradiance_spectrum", "thuillier_2003")
      # second-kind call to ``open()``
      ds = data.open(path="spectra/thuillier_2003.nc")

.. admonition:: Valid data set categories

   .. list-table::
      :widths: 1 1

      * - :class:`absorption_cross_section_spectrum <eradiate.data.absorption_spectra>`
        - Absorption cross section spectrum
      * - :class:`thermoprops_profiles <eradiate.data.thermoprops_profiles>`
        - Atmospheric thermophysical properties profiles
      * - :class:`solar_irradiance_spectrum <eradiate.data.solar_irradiance_spectra>`
        - Solar irradiance spectrum
      * - :class:`spectral_response_function <eradiate.data.spectral_response_function>`
        - Spectral response function

"""

import os

import xarray as xr

from .absorption_spectra import _AbsorptionGetter
from .chemistry import _ChemistryGetter
from .solar_irradiance_spectra import _SolarIrradianceGetter
from .spectral_response_function import _SpectralResponseFunctionGetter
from .thermoprops_profiles import _ThermoPropsProfilesGetter
from .. import path_resolver as _presolver

_getters = {
    "absorption_spectrum": _AbsorptionGetter,
    "chemistry": _ChemistryGetter,
    "thermoprops_profiles": _ThermoPropsProfilesGetter,
    "solar_irradiance_spectrum": _SolarIrradianceGetter,
    "spectral_response_function": _SpectralResponseFunctionGetter,
}


def open(category=None, id=None, path=None):
    """
    Open a data set.

    Parameter ``category`` (str or None):
        If ``None``, ``path`` must not be ``None`` .
        Dataset category identifier. Valid data set categories are listed in the
        documentation of the :mod:`~eradiate.data` module.

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

    Raises → OSError:
        I/O error while processing a handled resource.
    """
    if path is None:
        if category is None or id is None:
            raise ValueError("if 'path' is None, 'category' and 'id' must not be None")

        try:
            return getter(category).open(id)
        except ValueError:
            raise
        except OSError:
            raise

    # path is not None: we open the data
    fname = _presolver.resolve(path)
    ext = os.path.splitext(fname)[1]

    if ext == ".nc":
        return xr.open_dataset(fname)

    raise ValueError(f"cannot load resource {fname}")


def getter(category):
    """
    Get the getter class for the requested category.

    Parameter ``category`` (str):
        Dataset category identifier. See :func:`open` for valid categories.

    Returns → :class:`.DataGetter`:
        Getter class for the requested category.

    Raises → ValueError:
        Unknown requested category.
    """
    try:
        return _getters[category]
    except KeyError:
        raise ValueError(
            f"invalid data category '{category}'; must be one of "
            f"{list(_getters.keys())}"
        )


def registered(category):
    """
    Get a list of registered dataset IDs for a given data set category.

    Parameter ``category`` (str):
        Dataset category identifier. See :func:`open` for valid categories.

    Returns → list[str]:
        List of registered data set IDs for the selected category.
    """
    return getter(category).registered()


def find(category):
    """
    Check if the data referenced for a given category exists.

    Parameter ``category`` (str):
        Dataset category identifier. See :func:`open` for valid categories.

    Returns → dict[str, bool]:
        Report dictionary containing data set identifiers as keys and Boolean
        values (``True`` if a file exists for this ID, ``False`` otherwise).
    """
    return getter(category).find()
