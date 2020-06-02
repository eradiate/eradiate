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
        Learn about Eradiate, how to get it and how to compile it.
    :ref:`User guide<sec-user_guide-intro>`
        Learn how to use Eradiate's applications and API.
    :ref:`Developer guide<sec-developer_guide-intro>`
        Learn how to work with Eradiate's source code and modify it.
    :ref:`API reference<sec-api_reference-intro>`
        The complete API reference.


About
-----

Eradiate's development is funded by a European Space Agency project funded by the European Commission's Copernicus programme. The design phase was funded by the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy, Yvan Nollet and Sebastian Schunke.

Eradiate uses as its computational kernel a modified copy of the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ renderer. The Eradiate team acknowledges all Mitsuba 2 contributors for their exceptional work.

Contents
--------

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    Introduction<rst/getting_started/intro>
    rst/getting_started/getting_code
    rst/getting_started/building

.. toctree::
    :maxdepth: 2
    :caption: User guide

    Introduction<rst/user_guide/intro>
    rst/user_guide/post_processing

.. toctree::
    :maxdepth: 2
    :caption: Developer guide

    Introduction<rst/developer_guide/intro>
    rst/developer_guide/documentation
    rst/developer_guide/testing
    rst/developer_guide/plugin_development

.. toctree::
    :maxdepth: 2
    :caption: API reference

    Introduction<rst/api_reference/intro>
    rst/api_reference/kernel
    rst/api_reference/scenes
    rst/api_reference/solvers
    rst/api_reference/util

.. toctree::
    :maxdepth: 2
    :caption: Miscellaneous

    rst/miscellaneous/bibliography
    Todo list<rst/miscellaneous/todolist>
