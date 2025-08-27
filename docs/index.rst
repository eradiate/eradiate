:layout: simple

Eradiate Documentation
======================

**Date**: |today| |
**Version**: |version| |
:doc:`/release_notes/index`

**Useful links**:
`Home <https://www.eradiate.eu>`__ |
`Source repository <https://github.com/eradiate/eradiate>`__ |
`Issues & ideas <https://github.com/eradiate/eradiate/issues>`__ |
`Q&A support <https://github.com/eradiate/eradiate/discussions>`__

**Docs versions**:
`stable <https://eradiate.readthedocs.io/en/stable/>`__ |
`latest <https://eradiate.readthedocs.io/en/latest/>`__

Eradiate is a modern radiative transfer simulation software package written in
Python and C++17. It relies on a radiometric kernel based on the
`Mitsuba 3 <https://github.com/mitsuba-renderer/mitsuba3>`__ rendering system
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

   user_guide/index
   tutorials/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Reference

   reference_api/index
   reference_plugins/index
   reference_cli.md
   release_notes/index.md
   bibliography

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Data

   Introduction <data/intro>
   Atmosphere: Aerosols / particles <data/aerosols_particles>
   Atmosphere: Molecular absorption <data/absorption_databases>
   Atmosphere: Thermophysical properties <data/atmosphere_thermoprops>
   Solar irradiance <data/solar_irradiance>
   Spectral response functions <data/srf>
   RAMI benchmark scenes <data/rami_scenes>

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Developers/contributors

   contributing
   dependencies
   maintainer_guide
   developer_guide/index
