.. _sec-user_guide-data-absorption:

Absorption cross section
========================

Absorption cross section data sets provide the monochromatic absorption cross
section spectrum of a given absorbing species at specific pressure and
temperature conditions.

Data access
-----------

.. admonition:: Important
   :class: warning

   Most absorption cross section data used internally by the Eradiate team
   cannot be distributed easily due to their large size (around 1TB). This, in
   particular, means that the support for monochromatic simulation is currently
   limited to selected atmospheric profiles. We plan to improve the delivery of
   those spectra in the future.

Distributed data sets are managed the data store (see
:ref:`sec-user_guide-data-intro` for details).

Identifier format
^^^^^^^^^^^^^^^^^

Identifiers for absorption cross section data sets
are constructed based on the format ``{absorber}-{engine}-{wmin}_{wmax}`` where:

* ``absorber`` is the name of the absorber (*e.g.* ``CH4``),
* ``engine`` indicates the absorption cross section engine used (*e.g.* ``spectra``),
* ``wmin`` is the minimum wavenumber value in ``cm^-1``,
* ``wmax`` is the maximum wavenumber value in ``cm^-1``.

Structure
---------

Absorption cross section data sets include one data variable:

* monochromatic absorption cross section (``xs``)

one optional data variable:

* mixing ratios (``mr``),

three :term:`dimension coordinates <dimension coordinate>`:

* wavenumber (``w``),
* pressure (``p``),
* temperature (``t``),

(temperature may be a :term:`non-dimension coordinate`) and one optional
dimension coordinate:

* molecules (``m``).

Data variable ``xs`` is tabulated with respect to ``w``, ``p`` and ``t``
(optional).

Data variable ``mr``, when present, is tabulated with respect to ``m``.
Data variable ``mr`` should be included in data sets where the absorber is a
mixture of different absorbing molecules.

.. toctree::
   :hidden:

   spectra-us76_u86_4
