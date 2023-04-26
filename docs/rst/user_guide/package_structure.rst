.. _sec-user_guide-package_structure:

Package structure overview
==========================

This section documents the general principles underpinning the organisation of
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
    kernel Mitsuba.

Data handling and visualisation
-------------------------------

Data handling [:mod:`eradiate.data`]
    This package serves data shipped with Eradiate.

Post-processing pipeline definitions [:mod:`eradiate.pipelines`]
    This package provides a post-processing pipeline framework used to convert
    raw sensor results yielded by kernel sensors to quantities of interest for
    Earth observation applications (*e.g.* reflectance). The data is stored as
    xarray labelled arrays (:term:`Dataset`).

xarray utility functions [:mod:`eradiate.xarray`]
    Various support components taking advantage of the xarray library.

Plotting [:mod:`eradiate.plot`]
    This module defines optional utility functions to create
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

Numerical constants [:mod:`eradiate.constants`]
    Various numerical constants used throughout the code base.

Other support components
------------------------

Unit support [:mod:`eradiate.units`]
    Various utility functions and data variables used to safely handle unit
    conversions in Eradiate.

Math support [``eradiate.{`` :mod:`~eradiate.frame`, :mod:`~eradiate.quad`, :mod:`~eradiate.warp` ``}``]
    For the cases where Eradiate's math dependencies and kernel are not
    enough, additional mathematical tools are provided.

Random number generation [:mod:`eradiate.rng`]
    Support components for fine control of random number generation.

Exceptions [:mod:`eradiate.exceptions`]
    Exception and warning definitions.

Class writing facilities [``eradiate.{`` :mod:`~eradiate.attrs`, :mod:`~eradiate.converters`, :mod:`~eradiate.validators`, :mod:`~eradiate._factory` ``}``]
    These components are part of Eradiate's core class writing system. It relies
    on the `attrs <https://www.attrs.org>`_ library, extended for `Pint <https://pint.readthedocs.io>`_
    compatibility by the `Pinttrs <https://pinttrs.readthedocs.io>`_ library.

Miscellaneous [:mod:`eradiate.util`]
    Other support components which don't fit in any of the aforementioned
    classification entries.
