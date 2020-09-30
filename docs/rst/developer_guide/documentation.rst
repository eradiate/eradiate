.. _sec-developer_guide-documentation:


Building the documentation
==========================

Eradiate's documentation consists of two separate documents:

- the current document;
- the Mitsuba kernel documentation.

Building this document
----------------------

Once Eradiate is installed, this document can be built using the following commands:

.. code-block:: bash

    cd $ERADIATE_DIR/docs
    make html

After the build is completed, the html document is located in :code:`$ERADIATE_DIR/docs/_build/html`.

.. _sec_mitsuba_docs:

Building the kernel documentation
---------------------------------

Eradiate's Mitsuba kernel exposes a CMake target to build its documentation, which can be accessed once CMake is set up, as described in :ref:`sec-getting_started-building-mitsuba`.

Run the follwing commands:

.. code-block:: bash

    cd $ERADIATE_DIR/build
    ninja mkdoc

The compiled html documentation will then be located in :code:`$ERADIATE_DIR/build/html`.
