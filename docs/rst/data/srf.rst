.. _sec-data-srf:

Spectral response functions
===========================

A spectral response function dataset provides tabulated values of the spectral
response of a radiometric instrument.
Eradiate's built-in particle SRF datasets are managed by the data store
(see :ref:`sec-data-intro` for details).

Format
------

* **Format** ``xarray.Dataset`` (in-memory), NetCDF (storage)
* **Dimensions**

  * ``w``: radiation wavelength

* **Coordinates** (all dimension coordinates; when relevant, ``units`` are
  required and specified in the units metadata field)

  * ``w`` float [length]

* **Data variables** (when relevant, units are required and specified in the
  units metadata field)

  * ``srf`` (``w``): spectral response function [dimensionless]
  * ``srf_u`` (``w``): uncertainty on the ``srf`` data variable [dimensionless]

.. dropdown:: Full validation schema

   .. literalinclude:: /resources/data_schemas/srf_dataset_v1.yml

Naming convention
^^^^^^^^^^^^^^^^^

SRF data files are usually named ``{platform}-{instrument}-{band}.nc`` where:

* ``platform`` identifies the platform (*e.g.* satellite's name);
* ``instrument`` identifies the instrument onboard the platform;
* ``band`` specifies the spectral band.

For example, the spectral response function data set of the SLSTR instrument
onboard Sentinel-3B and in the spectral band 5 has the identifier
``sentinel_3b-slstr-5``.

Dataset index
-------------

.. important::

    The following datasets are processed to minimize the amount of
    computation performed for parts of the spectrum that will result
    in a low contribution to the final measure. Data are processed
    to keep the total integral of the SRF equal to at least 99.9%
    of that in the raw data. Raw data are also available, using a ``-raw`` suffix to the dataset ID (*e.g.* ``sentinel_3b-slstr-5-raw``).


``sentinel_2a-msi``
^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_2a-msi-1.nc``
* ``spectra/srf/sentinel_2a-msi-2.nc``
* ``spectra/srf/sentinel_2a-msi-3.nc``
* ``spectra/srf/sentinel_2a-msi-4.nc``
* ``spectra/srf/sentinel_2a-msi-5.nc``
* ``spectra/srf/sentinel_2a-msi-6.nc``
* ``spectra/srf/sentinel_2a-msi-7.nc``
* ``spectra/srf/sentinel_2a-msi-8.nc``
* ``spectra/srf/sentinel_2a-msi-9.nc``
* ``spectra/srf/sentinel_2a-msi-10.nc``
* ``spectra/srf/sentinel_2a-msi-11.nc``
* ``spectra/srf/sentinel_2a-msi-12.nc``
* ``spectra/srf/sentinel_2a-msi-8a.nc``


.. image:: /fig/srf/sentinel_2a-msi.png

``sentinel_2b-msi``
^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_2b-msi-1.nc``
* ``spectra/srf/sentinel_2b-msi-2.nc``
* ``spectra/srf/sentinel_2b-msi-3.nc``
* ``spectra/srf/sentinel_2b-msi-4.nc``
* ``spectra/srf/sentinel_2b-msi-5.nc``
* ``spectra/srf/sentinel_2b-msi-6.nc``
* ``spectra/srf/sentinel_2b-msi-7.nc``
* ``spectra/srf/sentinel_2b-msi-8.nc``
* ``spectra/srf/sentinel_2b-msi-9.nc``
* ``spectra/srf/sentinel_2b-msi-10.nc``
* ``spectra/srf/sentinel_2b-msi-11.nc``
* ``spectra/srf/sentinel_2b-msi-12.nc``
* ``spectra/srf/sentinel_2b-msi-8a.nc``


.. image:: /fig/srf/sentinel_2b-msi.png

``sentinel_3a-olci``
^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_3a-olci-1.nc``
* ``spectra/srf/sentinel_3a-olci-2.nc``
* ``spectra/srf/sentinel_3a-olci-3.nc``
* ``spectra/srf/sentinel_3a-olci-4.nc``
* ``spectra/srf/sentinel_3a-olci-5.nc``
* ``spectra/srf/sentinel_3a-olci-6.nc``
* ``spectra/srf/sentinel_3a-olci-7.nc``
* ``spectra/srf/sentinel_3a-olci-8.nc``
* ``spectra/srf/sentinel_3a-olci-9.nc``
* ``spectra/srf/sentinel_3a-olci-10.nc``
* ``spectra/srf/sentinel_3a-olci-11.nc``
* ``spectra/srf/sentinel_3a-olci-12.nc``
* ``spectra/srf/sentinel_3a-olci-13.nc``
* ``spectra/srf/sentinel_3a-olci-14.nc``
* ``spectra/srf/sentinel_3a-olci-15.nc``
* ``spectra/srf/sentinel_3a-olci-16.nc``
* ``spectra/srf/sentinel_3a-olci-17.nc``
* ``spectra/srf/sentinel_3a-olci-18.nc``
* ``spectra/srf/sentinel_3a-olci-19.nc``
* ``spectra/srf/sentinel_3a-olci-20.nc``
* ``spectra/srf/sentinel_3a-olci-21.nc``


.. image:: /fig/srf/sentinel_3a-olci.png

``sentinel_3a-slstr``
^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_3a-slstr-1.nc``
* ``spectra/srf/sentinel_3a-slstr-2.nc``
* ``spectra/srf/sentinel_3a-slstr-3.nc``
* ``spectra/srf/sentinel_3a-slstr-4.nc``
* ``spectra/srf/sentinel_3a-slstr-5.nc``
* ``spectra/srf/sentinel_3a-slstr-6.nc``
* ``spectra/srf/sentinel_3a-slstr-7.nc``
* ``spectra/srf/sentinel_3a-slstr-8.nc``
* ``spectra/srf/sentinel_3a-slstr-9.nc``


.. image:: /fig/srf/sentinel_3a-slstr.png

``sentinel_3b-olci``
^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_3b-olci-1.nc``
* ``spectra/srf/sentinel_3b-olci-2.nc``
* ``spectra/srf/sentinel_3b-olci-3.nc``
* ``spectra/srf/sentinel_3b-olci-4.nc``
* ``spectra/srf/sentinel_3b-olci-5.nc``
* ``spectra/srf/sentinel_3b-olci-6.nc``
* ``spectra/srf/sentinel_3b-olci-7.nc``
* ``spectra/srf/sentinel_3b-olci-8.nc``
* ``spectra/srf/sentinel_3b-olci-9.nc``
* ``spectra/srf/sentinel_3b-olci-10.nc``
* ``spectra/srf/sentinel_3b-olci-11.nc``
* ``spectra/srf/sentinel_3b-olci-12.nc``
* ``spectra/srf/sentinel_3b-olci-13.nc``
* ``spectra/srf/sentinel_3b-olci-14.nc``
* ``spectra/srf/sentinel_3b-olci-15.nc``
* ``spectra/srf/sentinel_3b-olci-16.nc``
* ``spectra/srf/sentinel_3b-olci-17.nc``
* ``spectra/srf/sentinel_3b-olci-18.nc``
* ``spectra/srf/sentinel_3b-olci-19.nc``
* ``spectra/srf/sentinel_3b-olci-20.nc``
* ``spectra/srf/sentinel_3b-olci-21.nc``


.. image:: /fig/srf/sentinel_3b-olci.png

``sentinel_3b-slstr``
^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/sentinel_3b-slstr-1.nc``
* ``spectra/srf/sentinel_3b-slstr-2.nc``
* ``spectra/srf/sentinel_3b-slstr-3.nc``
* ``spectra/srf/sentinel_3b-slstr-4.nc``
* ``spectra/srf/sentinel_3b-slstr-5.nc``
* ``spectra/srf/sentinel_3b-slstr-6.nc``
* ``spectra/srf/sentinel_3b-slstr-7.nc``
* ``spectra/srf/sentinel_3b-slstr-8.nc``
* ``spectra/srf/sentinel_3b-slstr-9.nc``


.. image:: /fig/srf/sentinel_3b-slstr.png

``aqua-modis``
^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/aqua-modis-1.nc``
* ``spectra/srf/aqua-modis-2.nc``
* ``spectra/srf/aqua-modis-3.nc``
* ``spectra/srf/aqua-modis-4.nc``
* ``spectra/srf/aqua-modis-5.nc``
* ``spectra/srf/aqua-modis-6.nc``
* ``spectra/srf/aqua-modis-7.nc``
* ``spectra/srf/aqua-modis-8.nc``
* ``spectra/srf/aqua-modis-9.nc``
* ``spectra/srf/aqua-modis-10.nc``
* ``spectra/srf/aqua-modis-11.nc``
* ``spectra/srf/aqua-modis-12.nc``
* ``spectra/srf/aqua-modis-13.nc``
* ``spectra/srf/aqua-modis-14.nc``
* ``spectra/srf/aqua-modis-15.nc``
* ``spectra/srf/aqua-modis-16.nc``


.. image:: /fig/srf/aqua-modis.png

``terra-modis``
^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/terra-modis-1.nc``
* ``spectra/srf/terra-modis-2.nc``
* ``spectra/srf/terra-modis-3.nc``
* ``spectra/srf/terra-modis-4.nc``
* ``spectra/srf/terra-modis-5.nc``
* ``spectra/srf/terra-modis-6.nc``
* ``spectra/srf/terra-modis-7.nc``
* ``spectra/srf/terra-modis-8.nc``
* ``spectra/srf/terra-modis-9.nc``
* ``spectra/srf/terra-modis-10.nc``
* ``spectra/srf/terra-modis-11.nc``
* ``spectra/srf/terra-modis-12.nc``
* ``spectra/srf/terra-modis-13.nc``
* ``spectra/srf/terra-modis-14.nc``
* ``spectra/srf/terra-modis-15.nc``
* ``spectra/srf/terra-modis-16.nc``
* ``spectra/srf/terra-modis-17.nc``
* ``spectra/srf/terra-modis-18.nc``
* ``spectra/srf/terra-modis-19.nc``
* ``spectra/srf/terra-modis-20.nc``
* ``spectra/srf/terra-modis-21.nc``
* ``spectra/srf/terra-modis-22.nc``
* ``spectra/srf/terra-modis-23.nc``
* ``spectra/srf/terra-modis-24.nc``
* ``spectra/srf/terra-modis-25.nc``
* ``spectra/srf/terra-modis-26.nc``


.. image:: /fig/srf/terra-modis.png

``jpss1-viirs``
^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/jpss1-viirs-i1.nc``
* ``spectra/srf/jpss1-viirs-i2.nc``
* ``spectra/srf/jpss1-viirs-i3.nc``
* ``spectra/srf/jpss1-viirs-i4.nc``
* ``spectra/srf/jpss1-viirs-i5.nc``
* ``spectra/srf/jpss1-viirs-m1.nc``
* ``spectra/srf/jpss1-viirs-m2.nc``
* ``spectra/srf/jpss1-viirs-m3.nc``
* ``spectra/srf/jpss1-viirs-m4.nc``
* ``spectra/srf/jpss1-viirs-m5.nc``
* ``spectra/srf/jpss1-viirs-m6.nc``
* ``spectra/srf/jpss1-viirs-m7.nc``
* ``spectra/srf/jpss1-viirs-m8.nc``
* ``spectra/srf/jpss1-viirs-m9.nc``
* ``spectra/srf/jpss1-viirs-m10.nc``
* ``spectra/srf/jpss1-viirs-m11.nc``
* ``spectra/srf/jpss1-viirs-m12.nc``
* ``spectra/srf/jpss1-viirs-m13.nc``
* ``spectra/srf/jpss1-viirs-m14.nc``
* ``spectra/srf/jpss1-viirs-m15.nc``
* ``spectra/srf/jpss1-viirs-m16.nc``
* ``spectra/srf/jpss1-viirs-m16a.nc``
* ``spectra/srf/jpss1-viirs-m16b.nc``


.. image:: /fig/srf/jpss1-viirs.png

``npp-viirs``
^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/npp-viirs-i1.nc``
* ``spectra/srf/npp-viirs-i2.nc``
* ``spectra/srf/npp-viirs-i3.nc``
* ``spectra/srf/npp-viirs-i4.nc``
* ``spectra/srf/npp-viirs-i5.nc``
* ``spectra/srf/npp-viirs-m1.nc``
* ``spectra/srf/npp-viirs-m2.nc``
* ``spectra/srf/npp-viirs-m3.nc``
* ``spectra/srf/npp-viirs-m4.nc``
* ``spectra/srf/npp-viirs-m5.nc``
* ``spectra/srf/npp-viirs-m6.nc``
* ``spectra/srf/npp-viirs-m7.nc``
* ``spectra/srf/npp-viirs-m8.nc``
* ``spectra/srf/npp-viirs-m9.nc``
* ``spectra/srf/npp-viirs-m10.nc``
* ``spectra/srf/npp-viirs-m11.nc``
* ``spectra/srf/npp-viirs-m12.nc``
* ``spectra/srf/npp-viirs-m13.nc``
* ``spectra/srf/npp-viirs-m14.nc``
* ``spectra/srf/npp-viirs-m15.nc``
* ``spectra/srf/npp-viirs-m16a.nc``
* ``spectra/srf/npp-viirs-m16b.nc``


.. image:: /fig/srf/npp-viirs.png

``metop_a-avhrr``
^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/metop_a-avhrr-1.nc``
* ``spectra/srf/metop_a-avhrr-2.nc``
* ``spectra/srf/metop_a-avhrr-3a.nc``
* ``spectra/srf/metop_a-avhrr-3b.nc``
* ``spectra/srf/metop_a-avhrr-4.nc``
* ``spectra/srf/metop_a-avhrr-5.nc``


.. image:: /fig/srf/metop_a-avhrr.png

``metop_b-avhrr``
^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/metop_b-avhrr-1.nc``
* ``spectra/srf/metop_b-avhrr-2.nc``
* ``spectra/srf/metop_b-avhrr-3a.nc``
* ``spectra/srf/metop_b-avhrr-3b.nc``
* ``spectra/srf/metop_b-avhrr-4.nc``
* ``spectra/srf/metop_b-avhrr-5.nc``


.. image:: /fig/srf/metop_b-avhrr.png

``metop_c-avhrr``
^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/metop_c-avhrr-1.nc``
* ``spectra/srf/metop_c-avhrr-2.nc``
* ``spectra/srf/metop_c-avhrr-3a.nc``
* ``spectra/srf/metop_c-avhrr-3b.nc``
* ``spectra/srf/metop_c-avhrr-4.nc``
* ``spectra/srf/metop_c-avhrr-5.nc``


.. image:: /fig/srf/metop_c-avhrr.png

``metop_sg-metimage``
^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/metop_sg-metimage-vii4.nc``
* ``spectra/srf/metop_sg-metimage-vii8.nc``
* ``spectra/srf/metop_sg-metimage-vii12.nc``
* ``spectra/srf/metop_sg-metimage-vii15.nc``
* ``spectra/srf/metop_sg-metimage-vii16.nc``
* ``spectra/srf/metop_sg-metimage-vii17.nc``
* ``spectra/srf/metop_sg-metimage-vii20.nc``
* ``spectra/srf/metop_sg-metimage-vii22.nc``
* ``spectra/srf/metop_sg-metimage-vii23.nc``
* ``spectra/srf/metop_sg-metimage-vii24.nc``
* ``spectra/srf/metop_sg-metimage-vii25.nc``
* ``spectra/srf/metop_sg-metimage-vii26.nc``
* ``spectra/srf/metop_sg-metimage-vii28.nc``
* ``spectra/srf/metop_sg-metimage-vii30.nc``
* ``spectra/srf/metop_sg-metimage-vii33.nc``
* ``spectra/srf/metop_sg-metimage-vii34.nc``
* ``spectra/srf/metop_sg-metimage-vii35.nc``
* ``spectra/srf/metop_sg-metimage-vii37.nc``
* ``spectra/srf/metop_sg-metimage-vii39.nc``
* ``spectra/srf/metop_sg-metimage-vii40.nc``


.. image:: /fig/srf/metop_sg-metimage.png

``mtg_i-fci``
^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/mtg_i-fci-nir13.nc``
* ``spectra/srf/mtg_i-fci-nir16.nc``
* ``spectra/srf/mtg_i-fci-nir22.nc``
* ``spectra/srf/mtg_i-fci-vis04.nc``
* ``spectra/srf/mtg_i-fci-vis05.nc``
* ``spectra/srf/mtg_i-fci-vis06.nc``
* ``spectra/srf/mtg_i-fci-vis08.nc``
* ``spectra/srf/mtg_i-fci-vis09.nc``


.. image:: /fig/srf/mtg_i-fci.png

``mtg_i-li``
^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/mtg_i-li-1.nc``
* ``spectra/srf/mtg_i-li-2.nc``


.. image:: /fig/srf/mtg_i-li.png

``proba_v-vegetation_left``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/proba_v-vegetation_left-blue.nc``
* ``spectra/srf/proba_v-vegetation_left-red.nc``
* ``spectra/srf/proba_v-vegetation_left-nir.nc``
* ``spectra/srf/proba_v-vegetation_left-swir.nc``


.. image:: /fig/srf/proba_v-vegetation_left.png

``proba_v-vegetation_center``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/proba_v-vegetation_center-blue.nc``
* ``spectra/srf/proba_v-vegetation_center-red.nc``
* ``spectra/srf/proba_v-vegetation_center-nir.nc``
* ``spectra/srf/proba_v-vegetation_center-swir.nc``


.. image:: /fig/srf/proba_v-vegetation_center.png

``proba_v-vegetation_right``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Datastore paths

* ``spectra/srf/proba_v-vegetation_right-blue.nc``
* ``spectra/srf/proba_v-vegetation_right-red.nc``
* ``spectra/srf/proba_v-vegetation_right-nir.nc``
* ``spectra/srf/proba_v-vegetation_right-swir.nc``


.. image:: /fig/srf/proba_v-vegetation_right.png
