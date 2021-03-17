.. _sec-user_guide-data-spectra_us76_u86_4:

``spectra-us76_u86_4``
======================

The ``spectra-us76_u86_4`` data set is an absorption cross section data set
for the the ``us76_u86_4`` absorbing gas mixture and computed using the
`SPECTRA <https://spectra.iao.ru>`_ Information System.

Description
-----------

Absorber
^^^^^^^^

The ``us76_u86_4`` absorber is a gas mixture defined by the mixing ratios
provided in the table below.

.. list-table::
   :widths: 2 1 1 1 1

   * - Species
     - N2
     - O2
     - CO2
     - CH4
   * - Mixing ratio [%]
     - 78.084
     - 20.9476
     - 0.0314
     - 0.0002

Each gas species in this gas mixture is an absorber.
The ``us76_u86_4`` absorber is named in this way because it corresponds to the
gas mixture defined by the ``us76`` atmosphere model
:cite:`NASA1976USStandardAtmosphere` in the region of altitudes below 86 km,
and restricted to the 4 main molecular species, namely N2, O2, CO2 and CH4.

Data source
^^^^^^^^^^^

The absorption cross sections data in this data set were computed using the
`SPECTRA <https://spectra.iao.ru>`_
Information System, which uses the HITRAN database (2016-edition)
:cite:`Gordon2016HITRAN2016MolecularSpectroscopic`.

Structure
^^^^^^^^^

The ``spectra-us76_u86_4`` data set includes two data variables:

* absorption cross section (``xs``)
* mixing ratios (``mr``)

and three
`dimension coordinates <http://xarray.pydata.org/en/stable/data-structures.html#coordinates>`_:

* wavenumber (``w``)
* pressure (``p``) and
* molecule (``m``).

The absorption cross section data variable is tabulated with respect to the
wavenumber and pressure coordinates.
The mixing ration data variable is tabulated with respect to the pressure and
molecule coordinates.

.. note::

   The data set does not include a temperature dimension coordinate because the
   temperature coordinate is mapped onto the pressure coordinate according to the
   pressure-temperature relationship defined by the ``us76`` thermophysical
   profile.
   This is why this data set is tied to the ``us76`` thermophysical profile.

Wavenumber coordinate
~~~~~~~~~~~~~~~~~~~~~
The wavenumber coordinate is a linearly spaced mesh between
:math:`4000 \, \mathrm{cm}^{-1}` (:math:`2500 \, \mathrm{nm}`) and
:math:`25711 \, \mathrm{cm}^{-1}` (:math:`390 \, \mathrm{nm}`)
with a constant wavenumber step of
:math:`0.00314 \, \mathrm{cm}^{-1}` (corresponding wavelength step varies
between
:math:`~ 0.00005 \, \mathrm{nm}` and
:math:`~ 0.002 \, \mathrm{nm}`).

Pressure coordinate
~~~~~~~~~~~~~~~~~~~

The pressure coordinate is a geometrically spaced mesh between
:math:`0.101325 \, \mathrm{Pa}` (:math:`10^{-6} \, \mathrm{atm}`) and
:math:`101325 \, \mathrm{Pa}`
with 64 mesh points.

The minimum pressure value of :math:`10^{-6} \, \mathrm{atm}` corresponds to a
maximum altitude of approximately 93 km, in the ``us76`` thermophysical profile.
This minimum value is a restriction of SPECTRA.
For higher altitudes, that is to say for lower pressure values, the absorption
cross section value is approximated to 0.

.. note::

   Given that the maximum value of the absorption cross section at 93 km is:

   .. math::

      \max_{\nu} \sigma_{a} = 9.62 \, 10^{-23} \, \mathrm{cm}^2,

   which corresponds to a maximal absorption coefficient value of:

   .. math::

      \max_{\nu} k_{a} = 4.0 \, 10^{-4} \, \mathrm{km}^{-1},

   this approximation seems reasonable.

Interpolation accuracy
----------------------

We assess the accuracy of the wavenumber and pressure interpolation of
the ``spectra-us76_u86_4`` absorption data set.

Method
^^^^^^

We down-sample the original data set on the corresponding axis, i.e. either
wavenumber or pressure, by selecting data corresponding to even-indices
coordinate values.
Then, we interpolate the down-sampled data set on the odd-indices coordinate
values.
We assess the interpolation accuracy by computing the relative errors:

.. math::

   \epsilon(p_i, q) =
   \frac{
      \lvert
      \sigma_{\mathrm{a, interpolated}}(p_i, q)
      - \sigma_{\mathrm{a, original}}(p_i, q)
      \rvert}{
      \sigma_{\mathrm{a, original}}(p_i, q)}

where:

* :math:`\sigma_{\mathrm{a, original}}` is the original data set absorption
  cross section,
* :math:`\sigma_{\mathrm{a, interpolated}}` is the interpolated down-sampled
  data set absorption cross section,
* :math:`p_i` is the interpolation coordinate (i.e. wavenumber or pressure)
  where :math:`i` is odd, and
* :math:`q` is the other coordinate (i.e. pressure or wavenumber).

.. note::
   Since the assessed interpolation accuracy here is that of the downsampled
   data set, we can expect that the interpolation accuracy of the original data
   set to be better.

We discard the cross section data where the following condition is met:

.. math::
   :label: negligible_k

   k_{\mathrm{a}}(\nu_{i}, p) < 10^{-3} \, \mathrm{cm}^{-1},

where :math:`k_{\mathrm{a}}(\nu_{i}, p)` is computed with:

.. math::

   k_{\mathrm{a}}(\nu_{i}, p) = n(p) \, \sigma_{\mathrm{a}}(\nu_{i}, p)

where :math:`n(p)` is the number density corresponding to the pressure :math:`p`
in the ``us76`` thermophysical profile.

We apply :eq:`negligible_k` because we estimate that these absorption coefficient
values are too small to influence radiative transfer computations significantly.

.. note::

   By interpolating the down-sampled absorption cross section data set at the
   odd-index coordinate values, we maximise the interpolation error
   with respect to the reference data set because each interpolation point is
   exactly at the middle of each data interval.
   This means that the relative errors are computed in the worst case scenarios,
   hence provide a conservative estimate of the interpolation accuracy.

Results
^^^^^^^

.. image:: fig/w_interp_rerr_histo_101325.png
   :align: center

.. image:: fig/p_interp_rerr_histo_65349.png
   :align: center

Analysis
^^^^^^^^

Wavenumber interpolation
~~~~~~~~~~~~~~~~~~~~~~~~

We make the following observations:

* At standard pressure (ground level altitude), the interpolation accuracy is
  relatively good -- better than 2 % except for some outliers.
* Up to altitudes ~ 10 km the interpolation accuracy remains fine -- better than
  5 % except for outliers.
* For pressure corresponding to 10 km altitude and higher, the interpolation
  accuracy gets poorer, although the errors counts decrease at the same time.
* As the pressure decreases (corresponding altitude increases), the interpolation
  accuracy gets poorer and poorer.

.. note::

   The counts number decreases with decreasing pressure because lower pressure
   means lower number density hence fewer absorption cross section data points
   satisfy :eq:`negligible_k`.

Pressure interpolation
^^^^^^^^^^^^^^^^^^^^^^

We make the following observations:

* The interpolation accuracy is relatively bad -- around 100 % -- at ground level
  altitude.

.. note::
   We plan to generate a new version of the data set with a finer pressure mesh
   to address this problem.
