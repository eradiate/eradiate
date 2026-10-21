Particle profile (Ppr)
======================

.. _sec-data_formats-ppr_core_v1:

Ppr-Core v1 [``ppr_core_v1``]
-----------------------------

This format defines a voxellized description of the :term:`state variables
<state variable>` of a particulate medium, :term:`amount` included. Each data
variable describes a :term:`field`, *i.e.* a spatially varying quantity, on 3D
gridded coordinates. Data variables are not prescribed:
a :class:`.ParticleStateComponent` binds them by name to the state dimensions of
a particle properties dataset and to the amount scaling its extinction.

The grid is described by the ``x_edges``/``y_edges``/``z_edges`` coordinates,
which hold the strictly increasing cell edges along each axis. Each is a
coordinate variable in the NetCDF sense (a 1-D variable named after its own
dimension), which xarray loads as a dimension coordinate. The profile is
defined on either a Cartesian coordinate system (``x_edges``/``y_edges`` in
length units) or a spherical geocentric coordinate system
(``x_edges``/``y_edges`` in angle units, representing azimuth/colatitude).

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``x``: cell index along the x axis
    * ``y``: cell index along the y axis
    * ``z``: cell index along the z (altitude) axis
    * ``x_edges``, ``y_edges``, ``z_edges``: cell edge index along each axis,
      of size ``len(x) + 1``, ``len(y) + 1`` and ``len(z) + 1`` respectively

Coordinates
    *Units are required and specified in the "units" metadata field.*

    * ``x_edges(x_edges)`` float [length or angle]: grid cell edges along the
      x axis
    * ``y_edges(y_edges)`` float [length or angle, same as ``x_edges``]: grid
      cell edges along the y axis
    * ``z_edges(z_edges)`` float [length]: grid cell edges along the z
      (altitude) axis

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * At least one ``<name>(z, y, x)`` float [any]: state variable field.
      The variable used as the amount is zero where no particles are present;
      other variables are then irrelevant and may take any value.

.. _sec-data_formats-ppr_psd_v1:

Ppr-PSD v1 [``ppr_psd_v1``]
---------------------------

Extension of ``ppr_core_v1`` describing a particle profile by its particle size
distribution (PSD) parameters and mass concentration. It is the profile
counterpart of :ref:`Prt-PSD v1 <sec-data_formats-prt_psd_v1>` and the format
expected by :class:`.ParticlePSDComponent` with its default bindings.

Data variables
    In addition to the ``ppr_core_v1`` variables:

    * ``reff(z, y, x)`` float [length]: effective radius of the particle
      size distribution
    * ``veff(z, y, x)`` float [—]: effective variance of the particle size
      distribution
    * ``mass_concentration(z, y, x)`` float [mass / volume]: particle mass
      concentration. Zero where no particles are present; ``reff``/``veff``
      are then irrelevant and may take any value.
