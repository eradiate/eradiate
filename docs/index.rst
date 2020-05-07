Eradiate: A New-generation Radiative Transfer Model for the Earth Observation Community
=======================================================================================

.. only:: not latex

    .. image:: fig/eradiate-logo-dark-no_bg.png
        :width: 75%
        :align: center

Eradiate is a radiative transfer simulation software package written in Python and C++17. It relies on a computational kernel based on the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ rendering system :cite:`Nimier-David2019MitsubaRetargetableForward`.

Eradiate uses Monte Carlo ray tracing integration methods to compute radiative transfer in scenes consisting of an arbitrary 3D geometry illuminated by an arbitrary number of light sources, possibly accounting for polarisation.

.. admonition:: Where should I go?

    :ref:`Getting started<sec-getting_started-intro>`
        Go here to learn about Eradiate, how to get it and how to compile it.
    :ref:`User guide<sec-user_guide-intro>`
        Go here to learn about how to use Eradiate's applications and API.
    :ref:`Developer guide<sec-developer_guide-intro>`
        Go here to learn about how to modify Eradiate.
    :ref:`API reference<sec-api_reference-intro>`
        Go here for the complete API reference.


About
-----

Eradiate's development is funded by a European Space Agency project funded by the European Commission's Copernicus programme. The design phase was funded by the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy, Yvan Nollet and Sebastian Schunke.

Eradiate inherits its core infrastructure from the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ renderer and the Eradiate team acknowledges all Mitsuba 2 contributors for their exceptional work.

Contents
--------

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    rst/getting_started/intro
    rst/getting_started/getting_code
    rst/getting_started/building

.. toctree::
    :maxdepth: 1
    :caption: User guide

    rst/user_guide/intro
    rst/user_guide/post_processing

.. toctree::
    :maxdepth: 2
    :caption: Developer guide

    rst/developer_guide/intro

.. toctree::
    :maxdepth: 1
    :caption: API reference

    rst/api_reference/intro
    rst/api_reference/kernel
    rst/api_reference/scenes
    rst/api_reference/solvers

.. toctree::
    :maxdepth: 1
    :caption: Miscellaneous

    rst/miscellaneous/bibliography
