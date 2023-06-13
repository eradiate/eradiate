Eradiate Documentation
======================

**Date**: |today| |
**Version**: |version| |
:doc:`src/CHANGELOG`

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

   .. grid-item-card:: :fas:`download` User Guide
      :link: sec-user_guide
      :link-type: ref

      Learn about Eradiate, how to get it and how to use it.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: sec-tutorials
      :link-type: ref

      A practical introduction to Eradiate.

   .. grid-item-card:: :fas:`file-code` Reference
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
   src/CHANGELOG.md
   rst/bibliography

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Developers/contributors


   rst/dependencies
   rst/contributing
   rst/maintainer_guide
   rst/developer_guide/index
