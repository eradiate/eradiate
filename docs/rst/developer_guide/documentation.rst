Documentation
=============

Building this document
----------------------

Once Eradiate is installed, this document can be built with only one step:

.. code-block:: bash

    python -m sphinx html docs build/html

After the build is completed, the html document is located in :code:`build/html`

.. _sec_mitsuba_docs:

Building the Mitsuba documentation
----------------------------------

Mitsuba exposes a CMake target to build its documentation, which can be accessed
once CMake is set up, as described in section :ref:`sec_compiling_mitsuba`.

Simply run the follwing command in Mitsuba'S build directory:

.. code-block:: bash

    ninja mkdoc

The html documentation will then be located in :code:`$MITSUBA_DIR/build/html`.