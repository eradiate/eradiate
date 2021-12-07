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

Core support [:mod:`eradiate`]
    The top-level module contains basic support functions required to do almost
    anything with Eradiate.

Scene generation [:mod:`eradiate.scenes`]
    This package exposes the scene generation components. It includes convenient
    interfaces to create objects representing the geometry, optical properties,
    illumination and measures in the scene on which you'll run your radiative
    transfer simulation.

Experiments [:mod:`eradiate.experiments`]
    In this package, you will find interfaces to trigger simulation runs,
    including pre- and post-processing operations.

Radiometric kernel
------------------

Kernel [:mod:`eradiate.kernel`]
    This module provides functionality related with Eradiate's radiometric
    kernel (a modified version of the Mitsuba 2 rendering system).

Data handling and visualisation
-------------------------------

Data handling [:mod:`eradiate.data`]
    This package serves data used by Eradiate. This data can be shipped by
    Eradiate, but also extended by users.

Post-processing pipeline definitions [:mod:`eradiate.pipelines <sec-reference-pipelines>`]
    This package provides a post-processing pipeline framework used to convert
    raw sensor results yielded by kernel sensors to quantities of interest for
    Earh observation applications (*e.g.* reflectance). The data is stored as
    xarray labelled arrays (:term:`Dataset`).

Plotting [:mod:`eradiate.plot`]
    This module defines utility functions to create
    `Matplotlib <https://matplotlib.org>`_-based visualisations of Eradiate's
    input and output data.

Physical properties
-------------------

Radiative properties [:mod:`eradiate.radprops`]
    This package provides abstractions used to define radiative properties used
    to create scenes.

Thermosphysical properties [:mod:`eradiate.thermoprops`]
    This package provides abstractions used to define thermophysical properties
    of scene objects. The output of its components are generally used as input
    of components responsible for radiative property computation.

Other support components
------------------------

Class writing facilities [``eradiate.{`` :mod:`~eradiate.attrs`, :mod:`~eradiate.converters`, :mod:`~eradiate.validators`, :mod:`~eradiate._factory` ``}``]
    These components are part of Eradiate's core class writing system. It relies
    on the `attrs <https://www.attrs.org>`_ library, extended for `Pint <https://pint.readthedocs.io>`_
    compatibility by the `Pinttrs <https://pinttrs.readthedocs.io>`_ library.

Unit support [:mod:`eradiate.units`]
    Various utility functions and data variables used to safely handle unit
    conversions in Eradiate.

Math support [``eradiate.{`` :mod:`~eradiate.frame`, :mod:`~eradiate.quad`, :mod:`~eradiate.warp` ``}``]
    For the cases where Eradiate's math dependencies and kernel are not
    enough, additional mathematical tools are provided.

Exceptions [:mod:`eradiate.exceptions`]
    This module contains exception and warning definitions.

Miscellaneous [:mod:`eradiate._util`]
    This module contains other support components which don't fit in any of the
    aforementioned classification entries.
