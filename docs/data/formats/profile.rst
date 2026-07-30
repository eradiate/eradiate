Particle profile (Ppr)
=======================

.. _sec-data-formats-ppr_v1:

Ppr v1 [``ppr_v1``]
-------------------

Sparse, per-voxel description of a spatially varying particle field.
Typically used for clouds. Each entry describes one populated voxel: its
grid cell indices, the local particle size distribution parameters, and the
mass concentration used to scale optical properties looked up from a
:ref:`Prt v1 <sec-data-formats-prt_v1>` dataset.

The grid is described by the ``x_levels``/``y_levels``/``z_levels`` arrays of
cell edges along each axis. The profile is defined on either a Cartesian
coordinate system (``x_levels``/``y_levels`` in length units) or a spherical
geocentric coordinate system (``x_levels``/``y_levels`` in angle units,
representing azimuth/colatitude).

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``index``: flat, populated-voxel index

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``x_levels`` float, 1-D [length or angle]: grid cell edges along the x
      axis
    * ``y_levels`` float, 1-D [length or angle]: grid cell edges along the y
      axis
    * ``z_levels`` float, 1-D [length]: grid cell edges along the z
      (altitude) axis
    * ``i_x(index)`` int [—]: index of the populated cell along the x axis,
      0-based
    * ``i_y(index)`` int [—]: index of the populated cell along the y axis,
      0-based
    * ``i_z(index)`` int [—]: index of the populated cell along the z axis,
      0-based
    * ``reff(index)`` float [length]: effective radius of the particle size
      distribution
    * ``veff(index)`` float [—]: effective variance of the particle size
      distribution
    * ``mass_concentration(index)`` float [mass / volume]: particle mass
      concentration
