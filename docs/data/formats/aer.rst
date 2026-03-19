Aerosol / particles (Aer)
=========================

.. _sec-data-formats-aer_core_v2:

Aer-Core v2 [``aer_core_v2``]
-----------------------------

This data format is derived from libRadtran's aerosol data format. It features
an adaptive scattering angle (θ) grid that allows for an optimal positioning of
samples depending on the wavelength.

The number of angular samples can vary across wavelengths up to a maximum equal
to the size of the ``iangle`` dimension. The per-wavelength count is stored in
``nangles``; entries beyond ``nangles[iw]`` in ``theta``, ``mu``, and ``phase``
are NaN-padded.

Similarly, the number of Legendre coefficients can vary across wavelengths up to
a maximum equal to the size of the ``imom`` dimension. When ``pmom`` is present,
``nmom`` stores the per-wavelength count; entries beyond ``nmom[iw]`` in
``pmom`` are NaN-padded.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``phamat``: nonzero coefficients in the phase matrix
    * ``iangle``: angular data points
    * ``imom``: Legendre coefficients for the (1,1) phase matrix component

Coordinates
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)`` float [length]: wavelength
    * ``phamat(phamat)`` str [—]: row and column indices of the phase matrix coefficient
      (*e.g.* "11", "12", etc.)
    * ``theta(w, iangle)`` float [angle]: scattering angle θ; NaN-padded beyond
      ``nangles[iw]``
    * ``mu(w, iangle)`` float [—]: value of cos θ; NaN-padded beyond ``nangles[iw]``

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``ext(w)`` float [1 / length]: extinction coefficient per unit concentration
    * ``ssa(w)`` float [—]: single-scattering albedo
    * ``phase(phamat, w, iangle)`` float [1 / sr]: value of the phase matrix;
      NaN-padded beyond ``nangles[iw]``
    * ``nangles(w)`` int [—]: number of valid angular samples per wavelength;
      must be consistent with the NaN padding of ``theta``, ``mu``, and ``phase``
    * ``pmom(w, imom)`` float [—], optional: Legendre expansion coefficients of
      the (1,1) phase matrix element; NaN-padded beyond ``nmom[iw]``
    * ``nmom(w)`` int [—], required if ``pmom`` is present: number of valid
      Legendre coefficients per wavelength; must be consistent with the NaN
      padding of ``pmom``

    .. note::

        ``pmom`` stores only the Legendre expansion of the (1,1) element
        :math:`p_{11}`.  Legendre expansions of the other phase matrix
        components are not part of this format.

Normalization conventions
    The (1,1) element of the phase matrix :math:`p_{11} (\mu)` follows the
    4π-normalized convention, normalizing the µ-parametrized phase function to
    2:

    .. math::
        \frac{1}{4\pi} \int_{4\pi} p_{11}(\Omega)\, \mathrm{d}\Omega = 1
        \Longleftrightarrow
        \int_{-1}^{1} p_{11}(\mu)\, \mathrm{d}\mu = 2,

    where :math:`p_{11} (\mu)` is in sr⁻¹.

    The ``pmom`` variable stores the Legendre expansion coefficients
    :math:`p_l = (2l + 1) \xi_l`, where

    .. math::

        \xi_l = \frac{1}{2} \int_{-1}^{1} p_{11}(\mu)\, P_l(\mu)\, \mathrm{d}\mu,

    and :math:`P_l` is the Legendre polynomial of degree :math:`l`. The phase
    function is thus recovered as

    .. math::

        p_{11}(\mu) = \sum_{l=0}^{L} \mathtt{pmom}[l]\, P_l(\mu).

    For a normalized phase function, :math:`\mathtt{pmom}[0] = 1` and
    :math:`\mathtt{pmom}[1] = 3g`, where :math:`g` is the asymmetry parameter.

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
    - * Angular data points (maximum)
      * ``iangle``
      * ``nthetamax``
    - * Angular data points (per wavelength)
      * ``nangles``
      * ``ntheta``
    - * Legendre coefficients
      * ``imom``
      * ``nmommax``

.. _sec-data-formats-aer_v1:

Aer v1 (legacy) [``aer_v1``]
----------------------------

The original aerosol / particle single-scattering radiative property format.
This format uses dense phase matrix storage and a fixed ``mu`` grid. It was
replaced by the more storage-efficient Aer-Core v2 format. Data provided in this
format can be converted to the Aer-Core v2 format using the
:func:`.aer_v1_to_aer_core_v2` function.

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
    * ``phase(w, mu, i, j)`` float [1 / sr]: scattering phase matrix

Normalization conventions
    The intended normalization for the (1,1) element of the phase matrix is the
    same 4π-normalized convention used by Aer-Core v2:

    .. math::

       \int_{-1}^{1} p_{11} (\mu) \, \mathrm{d}\mu = 2

    In practice, datasets in this format may not conform to this convention.
    The ``phase_scale`` parameter of :func:`.aer_v1_to_aer_core_v2` can be used
    to rescale the phase function during conversion in order to enforce the
    correct normalization. It should be noted that this has no impact on the
    correctness of Monte Carlo simulations using the Mitsuba radiometric kernel,
    as Mitsuba-level phase function components all enforce internally a correct
    normalization of statistical distributions, including tabulated phase
    functions.

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
