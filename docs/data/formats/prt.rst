Particles (Prt)
===============


.. _sec-data_formats-prt_core_v1:

Prt-Core v1 [``prt_core_v1``]
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

The format can be extended with *state dimensions* (see
:ref:`sec-data_formats-prt_extensions`).


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

    * ``ext(w)`` float [any]: extinction coefficient per unit amount (see
      :ref:`sec-data_formats-prt_ext_units`)
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

.. list-table:: Mapping of dimensions in the Prt-Core v1 format to the libRadtran equivalent.
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

.. seealso::

   The full specification of the ``aer_core_v2`` format:
   :ref:`sec-data_formats-aer_core_v2`

.. _sec-data_formats-prt_ext_units:

Extinction units
^^^^^^^^^^^^^^^^

The ``ext`` variable holds the extinction coefficient per unit amount of
particles. Its units are not fixed by the format, but must be specified in the
"units" metadata field. The component that consumes the dataset sets the
constraint:

* :class:`.ParticleOTComponent` only uses the spectral ratio
  :math:`\mathtt{ext}(\lambda) / \mathtt{ext}(\lambda_\mathrm{ref})`, so any
  units are accepted.
* :class:`.ParticleStateComponent` multiplies ``ext`` by a profile amount
  variable (*e.g.* mass concentration in g/m³); their product must have the
  dimensionality of an inverse length (*e.g.* ``ext`` in m²/g).

.. _sec-data_formats-prt_extensions:

Extensions
^^^^^^^^^^

Prt-Core v1 can be extended with one or several *state dimensions* on which
optical properties are indexed (*e.g.* particle size distribution parameters,
relative humidity). State dimensions are inserted after ``w``: every variable
and coordinate indexed by ``w`` (``ext``, ``ssa``, ``phase``, ``nangles``,
``pmom``, ``nmom``, ``mu``, ``theta``) gains the state dimensions, in the same
order, right after ``w``.

Each state dimension ``d`` comes with a coordinate ``d(d)`` that must be
strictly increasing, but need not be regularly spaced. Its units are free, but
must be specified in the "units" metadata field (``dimensionless`` if
relevant). Data are sorted in ascending order of each state coordinate.

The :ref:`sec-data_formats-prt_psd_v1` and :ref:`sec-data_formats-prt_hum_v1`
formats are named extensions that fix the name, meaning and units of their
state dimensions. Datasets with other state dimensions are valid
extensions; they are created with :func:`.make_prt_v1`.

:class:`.ParticleProperties` loads datasets with any number of state
dimensions. Components set further constraints:
:class:`.ParticleOTComponent` requires a dataset without state dimension, and
:class:`.ParticleStateComponent` requires 1 or 2 state dimensions, which is the
number of state variables the ``particlephase`` kernel plugin interpolates
over.


.. _sec-data_formats-prt_psd_v1:

Prt-PSD v1 [``prt_psd_v1``]
---------------------------

Named extension of ``prt_core_v1`` for particle properties. This data format
indexes properties defined by ``prt_core_v1`` on two state dimensions,
``reff`` and ``veff``, the effective radius and effective variance of the
particle size distribution (PSD), respectively. It shares the same
normalization conventions as ``prt_core_v1``.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``reff``: effective radius of the particle size distribution
    * ``veff``: effective variance of the particle size distribution
    * ``phamat``: nonzero coefficients in the phase matrix
    * ``iangle``: angular data points
    * ``imom``: Legendre coefficients for the (1,1) phase matrix component

Coordinates
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)``: float [length]: wavelength
    * ``reff(reff)``: float [length]: effective radius of the particle size
      distribution
    * ``veff(veff)``: float [—]: effective variance of the particle size
      distribution
    * ``phamat(phamat)``: str [—]: row and column indices of the phase matrix coefficient
      (*e.g.* "11", "12", etc.)
    * ``theta(w, reff, veff, iangle)``: float [angle]: scattering angle θ; NaN-padded beyond
      ``nangles[iw, ireff, iveff]``
    * ``mu(w, reff, veff, iangle)``: float [—]: value of cos θ; NaN-padded beyond
      ``nangles[iw, ireff, iveff]``

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``ext(w, reff, veff)`` float [any]: extinction coefficient per unit amount (see
      :ref:`sec-data_formats-prt_ext_units`)
    * ``ssa(w, reff, veff)`` float [—]: single-scattering albedo
    * ``phase(phamat, w, reff, veff, iangle)`` float [1 / sr]: value of the phase matrix;
      NaN-padded beyond ``nangles[iw, ireff, iveff]``
    * ``nangles(w, reff, veff)`` int [—]: number of valid angular samples per wavelength,
      reff and veff; must be consistent with the NaN padding of ``theta``, ``mu``, and
      ``phase``
    * ``pmom(w, reff, veff, imom)`` float [—], optional: Legendre expansion coefficients
      of the (1,1) phase matrix element; NaN-padded beyond ``nmom[iw, ireff, iveff]``
    * ``nmom(w, reff, veff)`` int [—], required if ``pmom`` is present: number of valid
      Legendre coefficients per wavelength; must be consistent with the NaN
      padding of ``pmom``

.. note::

    * Data are sorted in ascending order of ``w``, ``reff``, ``veff`` and ``mu``
    * ``reff`` and ``veff`` must be strictly increasing, but need not be
      regularly spaced
    * ``phamat`` conventions (valid values, and their implication for
      polarization and particle shape) are the same as ``prt_core_v1``

.. _sec-data_formats-prt_hum_v1:

Prt-Hum v1 [``prt_hum_v1``]
---------------------------

Named extension of ``prt_core_v1`` for hygroscopic particles. This data format
indexes properties defined by ``prt_core_v1`` on one state dimension,
``hum``, the ambient relative humidity. It is typically used for aerosol
components whose optical properties depend on humidity through water uptake
(*e.g.* OPAC components). It shares the same normalization conventions as
``prt_core_v1``.

Format
    ``xarray.Dataset`` (in-memory), NetCDF (storage)

Dimensions
    * ``w``: radiation wavelength
    * ``hum``: relative humidity
    * ``phamat``: nonzero coefficients in the phase matrix
    * ``iangle``: angular data points
    * ``imom``: Legendre coefficients for the (1,1) phase matrix component

Coordinates
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``w(w)``: float [length]: wavelength
    * ``hum(hum)``: float [—]: relative humidity (*e.g.* in percent)
    * ``phamat(phamat)``: str [—]: row and column indices of the phase matrix coefficient
      (*e.g.* "11", "12", etc.)
    * ``theta(w, hum, iangle)``: float [angle]: scattering angle θ; NaN-padded beyond
      ``nangles[iw, ihum]``
    * ``mu(w, hum, iangle)``: float [—]: value of cos θ; NaN-padded beyond
      ``nangles[iw, ihum]``

Data variables
    *When relevant, units are required and specified in the "units" metadata field.*

    * ``ext(w, hum)`` float [any]: extinction coefficient per unit amount (see
      :ref:`sec-data_formats-prt_ext_units`)
    * ``ssa(w, hum)`` float [—]: single-scattering albedo
    * ``phase(phamat, w, hum, iangle)`` float [1 / sr]: value of the phase matrix;
      NaN-padded beyond ``nangles[iw, ihum]``
    * ``nangles(w, hum)`` int [—]: number of valid angular samples per wavelength
      and humidity; must be consistent with the NaN padding of ``theta``, ``mu``,
      and ``phase``
    * ``pmom(w, hum, imom)`` float [—], optional: Legendre expansion coefficients
      of the (1,1) phase matrix element; NaN-padded beyond ``nmom[iw, ihum]``
    * ``nmom(w, hum)`` int [—], required if ``pmom`` is present: number of valid
      Legendre coefficients per wavelength and humidity; must be consistent with
      the NaN padding of ``pmom``

.. note::

    * Data are sorted in ascending order of ``w``, ``hum`` and ``mu``
    * ``hum`` must be strictly increasing, but need not be regularly spaced
    * ``phamat`` conventions (valid values, and their implication for
      polarization and particle shape) are the same as ``prt_core_v1``

Creation
--------

Datasets in all Prt variants are created with :func:`.make_prt_v1`, which
inserts optional state dimensions (*e.g.* ``reff`` and ``veff``, ``hum``)
between ``w`` and the angular and moment dimensions. :func:`.make_prt_core_v1`,
:func:`.make_prt_psd_v1` and :func:`.make_prt_hum_v1` are shortcuts for each
variant.

Conversion
----------

The following conversion components are provided:

.. list-table:: List of scattering particle property converters
    :header-rows: 1

    - * Function
      * Source format
      * Target format
    - * :func:`.aer_v1_to_prt_core_v1`
      * :ref:`sec-data_formats-aer_v1`
      * :ref:`sec-data_formats-prt_core_v1`
    - * :func:`.libradtran_to_prt`
      * libRadtran / MOPSMAP
      * :ref:`sec-data_formats-prt_core_v1`, or
        :ref:`sec-data_formats-prt_psd_v1` / :ref:`sec-data_formats-prt_hum_v1`
        when the output varies over ``reff`` / ``hum``
    - * :func:`.concat_particles_veff`
      * :ref:`sec-data_formats-prt_psd_v1` (several)
      * :ref:`sec-data_formats-prt_psd_v1`
