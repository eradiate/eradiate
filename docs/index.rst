:layout: simple

Eradiate Documentation
======================

**Date**: |today| |
**Version**: |version| |
:doc:`src/release_notes/index`

**Useful links**:
`Home <https://www.eradiate.eu>`_ |
`Source repository <https://github.com/eradiate/eradiate>`_ |
`Issues & ideas <https://github.com/eradiate/eradiate/issues>`_ |
`Q&A support <https://github.com/eradiate/eradiate/discussions>`_

**Docs versions**:
`stable <https://eradiate.readthedocs.io/en/stable/>`_ |
`latest <https://eradiate.readthedocs.io/en/latest/>`_

Eradiate is a modern radiative transfer simulation software package written in
Python and C++17. It relies on a radiometric kernel based on the
`Mitsuba 3 <https://github.com/mitsuba-renderer/mitsuba3>`_ rendering system
:cite:`Jakob2022DrJit,Jakob2022Mitsuba3`.

.. grid:: 1 2 auto auto
   :gutter: 3

   .. grid-item-card:: :iconify:`material-symbols:download-2 height=1.5em` User Guide
      :link: sec-user_guide
      :link-type: ref

      Learn about Eradiate, how to get it and how to use it.

   .. grid-item-card:: :iconify:`material-symbols:school height=1.5em` Tutorials
      :link: sec-tutorials
      :link-type: ref

      A practical introduction to Eradiate.

   .. grid-item-card:: :iconify:`material-symbols:dns height=1.5em` Data guide
      :link: sec-data-intro
      :link-type: ref

      Information about data formats and shipped datasets.

   .. grid-item-card:: :iconify:`material-symbols:description height=1.5em` Reference
      :link: sec-reference_api
      :link-type: ref

      The complete reference.

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Users

   rst/user_guide/index
   tutorials/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Reference

   rst/reference_api/index
   rst/reference_plugins/index
   src/reference_cli.md
   src/release_notes/index.md
   rst/bibliography

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Data

   Introduction <rst/data/intro>
   Atmosphere: Aerosols / particles <rst/data/aerosols_particles>
   Atmosphere: Molecular absorption <rst/data/absorption_databases>
   Atmosphere: Thermophysical properties <rst/data/atmosphere_thermoprops>
   Solar irradiance <rst/data/solar_irradiance>
   Spectral response functions <rst/data/srf>
   RAMI benchmark scenes <rst/data/rami_scenes>

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Developers/contributors

   rst/contributing
   rst/dependencies
   rst/maintainer_guide
   rst/developer_guide/index
