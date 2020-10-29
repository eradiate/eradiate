"""Functions to compute monochromatic absorption.

.. admonition:: Absorption cross section data set format:

    The data structure is a :class:`~xarray.Dataset` with specific data
    variables, dimensions and data coordinates.

    Data variables:

    - ``xs``: absorption cross section [cm^2]

    If the absorber is a mixture, an additional data variable is required:

    - ``mr``: mixing ratio []

    The dimensions of ``xs`` should be one of the following:

    - ``w``
    - (``w``, ``p``)
    - (``w``, ``t``)
    - (``w``, ``p``, ``t``)

    What dimensions are included indicate the relationship between the current dataset to other datasets and allow to combine these datasets together.
    Datasets with identical dimensions can be combined together along those dimensions, provided the absorber is the same.
    For example, a dataset with the dimensions (``w``, ``p``) for the ``us76`` absorber can be combined with other ``us76`` datasets with dimensions (``w``, ``p``), along the ``p`` dimension.

    The dimension of ``mr`` is ``m``.

    Data coordinates:

    - ``m``: absorbing molecule(s) []
    - ``w``: wavenumber [cm^-1]
    - ``p``: pressure [Pa]
    - ``t``: temperature [K]

    All these data coordinates are required even, if the corresponding dimensions do not not appear in the ``xs`` and ``mr``.
    In the latter case however, the data coordinates must be renamed by adding a ``c`` to the data coordinate name.
    This is to indicate that the data coordinate is a non-dimension data coordinate.
    See the difference between dimension coordinate and non-dimension coordinate at http://xarray.pydata.org/en/stable/data-structures.html

    Attributes:

    - ``convention``
    - ``title``
    - ``history``
    - ``source``
    - ``references``
    - ``absorber``: name of the absorbing molecule/mixture

    The meaning of the first 5 attributes is explained in the C.F. 1.8
    convention (http://cfconventions.org/).
"""

import os
import numpy as np

from ....util.units import ureg

_Q = ureg.Quantity


def get_mixture_name(species_set):
    """Returns the name of a gas mixture corresponding to a set of individual
    gas species.

    Parameter ``species_names`` (set):
        Set of absorbing gas species names.

    Raises → ``ValueError``:
        If the mixture is unknown.

    Returns → str:
        Mixture name.
    """
    us76 = {'N2', 'O2', 'Ar', 'CO2', 'Ne', 'He', 'Kr', 'Xe', 'CH4', 'H2', 'O',
            'H'}
    if species_set == us76:
        return "us76"
    else:
        raise ValueError("Unknown gas mixture.")


@ureg.wraps(ret="m^-1", args=("nm", None, None), strict=False)
def sigma_a(wavelength=550., profile=None, path=None):
    """Computes the monochromatic absorption coefficient in the thermophysical
    conditions given by a thermophysical ``profile``.

    .. note::
        This function requires that an absorption cross section data set
        corresponding to the absorbing gas species or gas mixture is
        present in the resources/data/spectra directory, or at a custom
        location if ``path`` set.

    .. note::
        So far, this function only supports the U.S. Standard Atmosphere 1976
        profile and as a results is able to compute the absorption coefficient
        only for the gas mixture included in that profile.

    Parameter ``wavelength`` (float or array):
        Wavelength value [nm].

    Parameter ``profile`` (`~xr.Dataset`):
        Atmosphere's thermophysical profile.

        Default value: U.S. Standard Atmosphere 1976 profile with 50 regular
        layers between 0 and 100 km.

    Parameter ``path`` (str):
        Path to the absorption cross section data set.

    Returns → float or array:
        Absorption coefficient [m^-1].
    """

    # make default profile
    if profile is None:
        from ..thermophysics.us76 import make_profile
        profile = make_profile()

    # check that profile is supported
    if profile.attrs["title"] != "U.S. Standard Atmosphere 1976":
        raise NotImplementedError(
            "Only the U.S. Standard Atmosphere 1976 profile is supported so "
            "far.")

    # load the absorption cross section data set
    if path is None:
        dir_path = "resources/data/spectra"
        try:
            ds = get(os.path.join(dir_path, "narrowband_usa_mls.nc"))
        except ValueError:
            raise ValueError(f"The data set could not be opened. Did you "
                             f"download and place the narrowband_usa_mls.nc "
                             f"data set in the {dir_path} repository?")
    else:
        ds = get(path)
    da = ds["absorption_cross_section"]

    # identify gas mixture in atmospheric thermophysical profile
    mixture = get_mixture_name(set(profile.species.values))

    # approximate us76 mixture to usa_mls mixture
    # TODO: fix that rough approximation
    if "usa_mls" in da.species.values:
        mixture = "usa_mls"

    # check that the gas mixture in the thermophysical profile matches
    # the species dimension of the absorption cross section data array
    if mixture not in da.species.values:
        raise ValueError(f"The gas mixture in the absorption cross section "
                         f"data set ({da.species.values[0]}) does not include "
                         f"that in the atmosphere thermophysical profile ("
                         f"{mixture})")

    # Compute absorption coefficient
    pressures = profile.p.values
    wavenumber = _Q(1. / wavelength, "nm^-1").to("cm^-1").magnitude
    cross_section = _Q(np.squeeze(
        da.interp(
            pressure=pressures,
            kwargs=dict(fill_value=0.)
        ).interp(
            wavenumber=wavenumber,
            kwargs=dict(fill_value=0.)
        ).values
    ), "cm^2")
    number_density = _Q(profile.n_tot.values, profile.n_tot.units)
    sigma = (number_density * cross_section).to("m^-1").magnitude

    return sigma
