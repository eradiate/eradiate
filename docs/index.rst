Eradiate: A New-generation Radiative Transfer Model for the Earth Observation Community
=======================================================================================

Eradiate is a radiative transfer simulation software package written in C++17 and Python. It uses a set of core libraries borrowed from the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ rendering system, a series of C++ plugins and a number of Python library components and applications.

Eradiate implements Monte Carlo ray tracing integration methods with which it can compute radiative transfer in scenes consisting of an arbitrary 3D geometry illuminated by an arbitrary number of light sources, possibly accounting for polarisation.

**New users** may want to jump to the :ref:`gettingstarted` section to learn about *getting* and
*compiling* eradiate. The :ref:`tutorials` section holds a series of guides, some of which interactive in jupyter notebooks, to familiarize users with the capabilities of Eradiate. The :ref:`advanced_topics` section holds refrences for developers and contributors to the Eradiate 
project.

About
-----

Eradiate's development is funded by a European Space Agency project funded by the European Commission's Copernicus programme. The design phase was funded by the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy, Yvan Nollet and Sebastian Schunke.

Eradiate inherits its core infrastructure from the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ renderer and the Eradiate team acknowledges all Mitsuba 2 contributors for their exceptional work.

.. only:: not latex

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Introduction

    introduction/getting_started
    bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
