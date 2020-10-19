"""Atmospheric thermophysical field data sets definition and creation.

.. note::
    The data sets defined here do not represent the point values of the
    atmospheric thermophysical field but instead represent the field values in
    homogeneous cells of finite extent. The extent of each cell is described
    using additional data coordinates (one data coordinate per spatial
    dimension). These data coordinates are not associated with any dimension
    of the data variables in the atmospheric thermophysical field data sets.

.. admonition:: Atmospheric vertical profile data set specification

    The atmospheric vertical profile gives the atmospheric pressure,
    temperature, total number density and number density by species over a
    given altitude mesh. This altitude mesh comprises levels and layers. Levels
    and layers are the nodes and cells of the 1D altitude mesh, respectively.

    The atmospheric vertical profile data set is a :class:`~xarray.Dataset` with
    specific data variables, dimensions, data coordinates and metadata.

    Data variables are:

    - ``p``: layer pressure [Pa],
    - ``t``: layer temperature [K],
    - ``n_tot``: layer total number density [m^-3],
    - ``n``: layer number density of the individual species [m^-3].

    Dimensions are ``z_layer``, ``z_level`` and ``species``:

    - ``p``, ``t`` and ``n_tot`` have the dimension ``z_layer``.
    - ``n`` has the dimensions ``species`` and ``z_layer``.
    - No variable has the dimension ``z_level``.

    Data coordinates are:

    - ``z_layer``: layer altitude [m]. The layer altitude is an altitude representative of the given layer, e.g. the middle of the layer.
    - ``z_level``: level altitude [m]. The sole purpose of this data coordinate is to store the information on the layers sizes.
    - ``species``: gas species [dimensionless].

    The units and standard name of each data variables and coordinates must
    be specified. The table below indicates the standard names and units of
    the different variables in this data set.

    .. list-table:: Standard names
       :widths: 1 1 1
       :header-rows: 1

       * - Variable
         - Standard name
         - Units
       * - ``p``
         - air_pressure
         - Pa
       * - ``t``
         - air_temperature
         - K
       * - ``n``
         - number_density
         - m^-3
       * - ``n_tot``
         - air_number_density
         - m^-3
       * - ``z_layer``
         - layer_altitude
         - m
       * - ``z_level``
         - level_altitude
         - m
       * - ``species``
         - species
         -

    In addition, the data set must include the following metadata attributes:
    ``convention``, ``title``, ``history``, ``source`` and ``reference``.
    Please refer to the `NetCDF Climate and Forecast (CF) Metadata Conventions <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_
    for a description of these attributes.
    Additional attributes are allowed.

.. admonition:: Atmospheric 2D/3D field data set specification

    (coming).
"""

import eradiate.scenes.atmosphere.thermophysics.us76 as us76

VAR = ["p", "t", "n_tot", "n", "species", "z_layer", "z_level"]

REQUIRED_DATA_VARS = ["p", "t", "n_tot", "n"]

REQUIRED_DIMS = ["z_layer", "z_level", "species"]

REQUIRED_COORDS = ["z_layer", "z_level", "species"]

UNITS = {
    "p": "Pa",
    "t": "K",
    "n_tot": "m^-3",
    "n": "m^-3",
    "z_layer": "m",
    "z_level": "m",
    "species": ""
}

STANDARD_NAME = {
    "p": "air_pressure",
    "t": "air_temperature",
    "n": "number_density",
    "n_tot": "air_number_density",
    "z_layer": "layer_altitude",
    "z_level": "level_altitude",
    "species": "species"
}


def check_vertical_profile(profile):
    r"""Checks that a given data set is an atmospheric vertical profile.

    The function goes through the data variables and coordinates and check
    that all required variables are in the data set and have the correct
    units and standard names. The function then checks that the data set
    metadata includes all required fields.

    Parameter ``profile`` (:class:`~xarray.Dataset`):
        The atmospheric vertical profile to check.

    Raises → ValueError:
        If the passed profile does not match the required format for vertical
        profile.
    """

    from eradiate.util.xarray import check_var_metadata, check_metadata

    if not all(var in list(profile.data_vars) for var in REQUIRED_DATA_VARS):
        missing_vars = [
            v for v in REQUIRED_DATA_VARS if v not in list(profile.data_vars)
        ]
        raise ValueError(
            f"Not all required data variables are in the data sets. Missing "
            f"variables are: {missing_vars}"
        )

    if not len(profile.data_vars) == len(REQUIRED_DATA_VARS):
        raise ValueError(
            f"The number of data variables must be {len(REQUIRED_DATA_VARS)} "
            f"but was found to be {len(profile.data_vars)}"
        )

    if not all(coord in list(profile.coords) for coord in REQUIRED_COORDS):
        raise ValueError(
            f"Not all required data coordinates are in the data sets. Missing"
            f" variables are: "
            f"{[c for c in REQUIRED_COORDS if c not in list(profile.coords)]}"
        )

    if not len(profile.coords) == len(REQUIRED_COORDS):
        raise ValueError(
            f"The number of data coordinates must be {len(REQUIRED_COORDS)} "
            f"but was found to be {len(profile.coords)}")

    for name in VAR:
        check_var_metadata(profile, name, UNITS[name], STANDARD_NAME[name])

    check_metadata(profile)
