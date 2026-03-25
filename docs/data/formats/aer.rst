Aerosol / particles (Aer)
=========================

Aer-Core v2
-----------

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
    add an ``nangle`` variable indexed by the ``w`` dimension.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``phamat``: nonzero coefficients in the phase matrix
    * ``iangle``: angular data points
    * ``imom``: Legendre coefficients

Coordinates
    *All dimension coordinates; \
    when relevant, units are required and specified in the "units" metadata field.*

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

Aer v1
------
