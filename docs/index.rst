Eradiate Documentation
======================

Eradiate is a modern radiative transfer simulation software package written in
Python and C++17. It relies on a computational kernel based on the
`Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ rendering system
:cite:`Nimier-David2019MitsubaRetargetableForward`.

**Date**: |today| |
**Version**: |version| |
:doc:`src/CHANGELOG`

**Useful links**:
`Source repository <https://github.com/eradiate/eradiate>`_ |
`Issues & ideas <https://github.com/eradiate/eradiate/issues>`_ |
`Q&A support <https://github.com/eradiate/eradiate/discussions>`_

**Docs versions**:
`stable <https://eradiate.readthedocs.io/en/stable/>`_ |
`latest <https://eradiate.readthedocs.io/en/latest/>`_

.. grid:: 1 2 auto auto
   :gutter: 3

   .. grid-item-card:: :octicon:`download` Getting started
      :link: sec-getting_started
      :link-type: ref

      Learn about Eradiate, how to get it and how to compile it.

   .. grid-item-card:: :octicon:`mortar-board` Tutorials
      :link: sec-tutorials
      :link-type: ref

      A practical introduction to Eradiate.

   .. grid-item-card:: :octicon:`book` User guide
      :link: sec-user_guide
      :link-type: ref

      Learn how to use Eradiate.

   .. grid-item-card:: :octicon:`file-code` Reference
      :link: sec-reference_api
      :link-type: ref

      The complete reference.

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Users

   rst/getting_started/index
   rst/user_guide/index
   tutorials/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Reference

   rst/reference_api/index
   rst/reference_plugins/index
   rst/reference_cli/index
   src/CHANGELOG.md
   rst/bibliography

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Developers/contributors

   rst/contributing
   rst/maintainer_guide
   rst/developer_guide/index
