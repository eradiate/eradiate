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

import eradiate.data as data
from ....util.units import ureg


@ureg.wraps(ret="m^-1", args=("nm", None, None), strict=False)
def compute_sigma_a(wavelength=550., profile=None, dataset_id=None):
    """Computes the monochromatic absorption coefficient in the thermophysical
    conditions given by a thermophysical ``profile``.

    .. note::
        This function requires that the absorption cross section datasets
        corresponding to ``dataset_id`` are downloaded and placed at the
        appropriate location. See :mod:`~eradiate.data` module for more
        information about the registered dataset ids and the corresponding
        paths.

    .. note::
        So far, this function only supports the U.S. Standard Atmosphere 1976
        profile.

    Parameter ``wavelength`` (float or array):
        Wavelength value [nm].

    Parameter ``profile`` (`~xr.Dataset`):
        Atmosphere's thermophysical profile.

        Default value: U.S. Standard Atmosphere 1976 profile with 50 regular
        layers between 0 and 100 km.

    Parameter ``dataset_id`` (str):
        Dataset identifier.

        .. warning::

            This parameter serves as debugging tool.
            The dataset identifier is determined automatically, based upon the
            atmospheric thermophysical profile. Use only if you know
            what you are doing.

    Returns → float or array:
        Absorption coefficient [m^-1].
    """
    wavenumber = (1 / ureg.Quantity(wavelength, "nm")).to("cm^-1")

    # make default profile
    if profile is None:
        from ..thermophysics.us76 import make_profile
        profile = make_profile()

    n_tot = ureg.Quantity(profile.n_tot.values, profile.n_tot.units)
    p = ureg.Quantity(profile.p.values, profile.p.units)

    # open the absorption cross section dataset and interpolate
    if profile.attrs["title"] == "U.S. Standard Atmosphere 1976":
        if dataset_id is None:
            dataset_id = "us76_u86_4-fullrange"
        ds = data.open(category="absorption_spectrum", id=dataset_id)
        xsw = ds.xs.interp(w=wavenumber.magnitude)

        # interpolate dataset in pressure
        xsp = xsw.interp(
            p=p.magnitude,
            kwargs=dict(fill_value=0.)  # this is required to handle the
            # pressure values that are smaller than 0.101325 Pa (the pressure
            # point with the smallest value in the absorption datasets) in the
            # US76 profile. These small pressure values occur above the
            # altitude of 93 km. Considering that the air number density at
            # these altitudes is small than the air number density at the
            # surface by a factor larger than 1e5, we assume that the
            # corresponding absorption coefficient is negligible compared to
            # 0.01 km^-1.
        )
        xs = ureg.Quantity(xsp.values, xsp.units)
    else:
        raise NotImplementedError(
            "Only the U.S. Standard Atmosphere 1976 profile is supported so "
            "far.")

    sigma = (n_tot * xs).to("m^-1").magnitude

    return sigma
