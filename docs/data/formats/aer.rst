Aerosol / particles (Aer)
=========================

.. _sec-data-formats-aer_core_v2:

Aer-Core v2 [``aer_core_v2``]
-----------------------------

This data format is derived from libRadtran's aerosol data format. It features
an adaptive scattering angle (θ) grid that allows for an optimal positioning of
samples depending on the wavelength.

A key difference with the libRadtran format is that the number of angular samples
is independent of the wavelength. When converting libRadtran datasets, the
number of cos θ values is the maximum value found in the ``nangle`` variable.
This does not increase the in-memory size of the dataset.

.. note::
    The reason why the number of θ points is kept constant is that updating the
    number of angular samples during the simulation is inefficient.
    Should this feature be reintroduced, all that would be required would be to
    add an ``nangles`` variable indexed by the ``w`` dimension.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``phamat``: nonzero coefficients in the phase matrix
    * ``iangle``: angular data points
    * ``imom``: Legendre coefficients

Coordinates
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)`` float [length]: wavelength
    * ``phamat(phamat)`` str [—]: row and column indices of the phase matrix coefficient
      (*e.g.* "11", "12", etc.)
    * ``theta(w, iangle)`` float [angle]: scattering angle θ
    * ``mu(w, iangle)`` float [—]: value of cos θ

Data variables
    *When relevant, units are required and  specified in the "units" metadata field.*

    * ``ext(w)`` float [1 / length]: extinction coefficient per unit concentration
    * ``ssa(w)`` float [—]: single-scattering albedo
    * ``phase(phamat, w, iangle)`` float [1 / solid angle]: value of the phase
      function (integral normalized to 2)
    * ``nmom(w)`` int [—], optional: number of nonzero Legendre coefficients
      (all values in ``pmom`` beyond this index are nan)
    * ``pmom(phamat, w, imom)`` float [—], optional: values of Legendre coefficient

.. note::

    * Data are sorted in ascending order of ``w`` and ``mu``
    * Valid ``phamat`` values are (in that order):
      ``["11", "12", "33", "34", "22", "44"]``
    * The number of ``phamat`` values implicitly defines whether the data
      describes light polarization: 1 means without polarization, 2+ means with
      polarization
    * The number of ``phamat`` values implicitly defines the scattering particle
      shape: 1 or 4 means spherical particles, 6 means spheroidal particles

.. list-table:: Mapping of dimensions in the Aer-Core v2 format to the libRadtran equivalent.
    :header-rows: 1

    - *
      * Eradiate
      * libRadtran
    - * Spectral dimension
      * ``w``
      * ``nlam``
    - * Phase matrix coefficients
      * ``phamat``
      * ``phamat``
    - * Angular data points
      * ``iangle``
      * ``nthetamax``
    - * Legendre coefficients
      * ``imom``
      * ``nmommax``

.. _sec-data-formats-aer_v1:

Aer v1 (legacy) [``aer_v1``]
----------------------------

The original aerosol / particle single-scattering radiative property format.
This format uses dense phase matrix storage and a fixed ``mu`` grid. I was
replaced by the more storage-efficient Aer-Core v2 format. Data provided in this
format can be converted to the Aer-Core v2 format using the
:func:`aer_v1_to_aer_core_v2` function.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``mu``: scattering angle cosine
    * ``i``: scattering phase matrix row index
    * ``j``: scattering phase matrix column index

Coordinates
    *All dimension coordinates; \
    when relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)`` float [length]: wavelength
    * ``mu(mu)`` float [—]: scattering angle cosine
    * ``i(i)``, ``j(j)`` int [—]: phase matrix row and column indices

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``sigma_t(w)`` float [1 / length]: volume extinction coefficient
    * ``albedo(w)`` float [—]: single-scattering albedo
    * ``phase(w, mu, i, j)`` float [1 / solid angle]: scattering phase matrix

Conventions
    * Phase matrix coefficients use C-style indexing (from 0).

Conversion
----------

The following conversion components are provided:

.. list-table:: List of scattering particle property converters
    :header-rows: 1

    - * Function
      * Source format
      * Target format
    - * :func:`.aer_v1_to_aer_core_v2`
      * Aer v1
      * Aer-Core v2
    - * :func:`.libradtran_to_aer_core_v2`
      * libRadtran / MOPSMAP
      * Aer-Core v2
    - * :func:`.libradtran_to_aer_v1`
      * libRadtran / MOPSMAP
      * Aer v1
