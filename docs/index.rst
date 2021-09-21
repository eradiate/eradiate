Eradiate Documentation
======================

*A New-generation Radiative Transfer Model for the Earth Observation Community*

Eradiate is a radiative transfer simulation software package written in Python
and C++17. It relies on a computational kernel based on the
`Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ rendering system
:cite:`Nimier-David2019MitsubaRetargetableForward`.

Eradiate uses Monte Carlo ray tracing integration methods to compute radiative
transfer in scenes consisting of an arbitrary 3D geometry illuminated by an
arbitrary number of light sources, possibly accounting for polarisation.


Where Should I Go?
------------------

:ref:`Getting started<sec-getting_started>`
    Learn about Eradiate, how to get it and how to compile it.
:ref:`User guide<sec-user_guide>`
    Learn how to use Eradiate's applications and API.
:ref:`Developer guide<sec-developer_guide>`
    Learn how to work with Eradiate's source code and modify it.
:ref:`Reference<sec-reference>`
    The complete API reference.


About
-----

Eradiate's development is funded by a European Space Agency project funded by
the European Commission's Copernicus programme. The design phase was funded by
the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy,
Yvan Nollet, Sebastian Schunke and Nicolas Misk.

Eradiate uses as its computational kernel a modified copy of the
`Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ renderer.
The Eradiate team acknowledges all Mitsuba 2 contributors for their exceptional
work.

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: For users

   rst/getting_started/index
   rst/user_guide/index
   examples/generated/tutorials/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: For developers/contributors

   rst/developer_guide/index
   rst/reference/index

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   :caption: Other topics

   rst/miscellaneous/index
