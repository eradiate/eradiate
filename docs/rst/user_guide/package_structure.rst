.. _sec-user_guide-package_structure:

Package structure overview
==========================

This section documents the general principles underlying the organisation of
the Eradiate Python package.

.. only:: latex

   .. image:: ../../fig/package.png

.. only:: not latex

   .. image:: ../../fig/package.svg

Main entry points
-----------------

Core support [:ref:`eradiate <sec-reference-core>`]
    The top-level module contains basic support functions required to do almost
    anything with Eradiate.

Scene generation [:ref:`eradiate.scenes <sec-reference-scenes>`]
    This package exposes the scene generation components. It includes convenient
    interfaces to create objects representing the geometry, optical properties,
    illumination and measures in the scene on which you'll run your radiative
    transfer simulation.

Experiments [:ref:`eradiate.solvers <sec-reference-experiments>`]
    In this package, you will find interfaces to trigger simulation runs,
    including pre- and post-processing operations.

Low-level computational kernel
------------------------------

Kernel [:ref:`eradiate.kernel <sec-reference-kernel>`]
    This module provides low-level interface components to Eradiate's
    radiometric kernel (a modified version of the Mitsuba 2 rendering system).
    Currently, this package simply checks that the kernel is installed and
    can be imported. The ``experiment`` package is, under the hood, a high-level
    interface to the kernel components.

Data handling and visualisation
-------------------------------

Data handling [:ref:`eradiate.data <sec-reference-data>`]
    This package serves data used by Eradiate. This data can be shipped by
    Eradiate, but also extended by users.

Post-processing pipeline definitions [:ref:`eradiate.pipelines <sec-reference-pipelines>`]
    This package provides a post-processing pipeline framework used to convert
    raw sensor results yielded by kernel sensors to quantities of interest for
    Earh observation applications (*e.g.* reflectance). The data is stored as
    xarray labelled arrays (:term:`Dataset`).

Plotting [:ref:`eradiate.plot <sec-reference-plot>`]
    This module defines utility functions to create
    `matplotlib <https://matplotlib.org>`_-based visualisations of Eradiate's
    input and output data.

Physical properties
-------------------

Radiative properties [:ref:`eradiate.radprops <sec-reference-radprops>`]
    This package provides abstractions used to define radiative properties used
    to create scenes.

Thermosphysical properties [:ref:`eradiate.thermoprops <sec-reference-thermoprops>`]
    This package provides abstractions used to define thermophysical properties
    of scene objects. The output of its components are generally used as input
    of components responsible for radiative property computation.

Other support components
------------------------

Class writing facilities [:ref:`eradiate.{attrs, converters, validators, _factory} <sec-reference-class_writing>`]
    These components are part of Eradiate's core class writing system. It relies
    on the `attrs <https://www.attrs.org>`_ library, extended for `Pint <https://pint.readthedocs.io>`_
    compatibility by the `Pinttrs <https://pinttrs.readthedocs.io>`_ library.

Unit support [:ref:`eradiate.units <sec-reference-units>`]
    Various utility functions and data variables used to safely handle unit
    conversions in Eradiate.

Math support [:ref:`eradiate.{frame, quad, warp} <sec-reference-math>`]
    For the cases where Eradiate's math dependencies and kernel are not
    enough, additional mathematical tools are provided.

Exceptions [:ref:`eradiate.exceptions <sec-reference-exceptions>`]
    This module contains exception and warning definitions.

Miscellaneous [:ref:`eradiate._util <sec-reference-misc>`]
    This module contains support components which don't fit in any of the
    aforementioned classification entries.
