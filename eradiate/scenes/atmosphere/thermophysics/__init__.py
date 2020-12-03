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

from . import us76
