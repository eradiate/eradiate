.. _sec-developer_guide-conventions_practices:

Conventions and development practices
=====================================

This pages briefly explains a few conventions and practices in the Eradiate development team.

Style
-----

The Eradiate codebase is written following Python's `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_. Its code formatter of choice is `yapf <https://github.com/google/yapf>`_ and its import formatter of choice is `isort <https://pycqa.github.io/isort/>`_ (version 5 or later), for which configuration files are provided at the root of the project. We also use the PyCharm CE IDE and its built-in formatting facilities, which yield similar but slightly different results.

Angular dependencies and variable names
---------------------------------------

Angle naming in Earth observation and radiative transfer modelling may sometimes clash or be confusing. Eradiate clearly distinguishes between two types of angular dependencies of its variables:

* Physical properties such as BRDFs and phase function have bidirectional dependencies which are referred to as *incoming* and *outgoing* directions. Datasets representing such quantities use  coordinate variables ``theta_i``, ``phi_i`` for the incoming direction's azimuth and zenith angles, and ``theta_o``, ``phi_o`` for their outgoing counterparts.

* Observations are usually parametrised by *illumination* and *viewing* directions. For datasets representing such results, Eradiate uses coordinate variables ``sza``, ``saa`` for *sun zenith/azimuth angle* and ``vza``, ``vaa`` for *viewing zenith/azimuth angle*.

.. note::

   In specific circumstances, one can directly convert an observation dataset to a physical property dataset. This, for instance, applies to top-of-atmosphere BRF data, but also any BRF computed or measured on a vaccuum. In such cases, incoming/outgoing directions can be directly converted to illumination/viewing directions. But in general, this does not work.
