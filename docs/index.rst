Eradiate: A New-generation Radiative Transfer Model for the Earth Observation Community
=======================================================================================

.. only:: not latex

    .. image:: fig/eradiate-logo-dark-no_bg.png
        :width: 75%
        :align: center

Eradiate is a radiative transfer simulation software package written in Python and C++17. It relies on a computational kernel based on the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ rendering system :cite:`Nimier-David2019MitsubaRetargetableForward`.

Eradiate uses Monte Carlo ray tracing integration methods to compute radiative transfer in scenes consisting of an arbitrary 3D geometry illuminated by an arbitrary number of light sources, possibly accounting for polarisation.

.. admonition:: Where do I go?

   **New users** may want to jump to the :ref:`Getting started<sec-getting_started-intro>` section to learn about how to get and compile Eradiate. The :ref:`Tutorials<sec-tutorials-intro>` section holds a series of guides, some of which interactive in jupyter notebooks, to familiarize users with the capabilities of Eradiate. The :ref:`Advanced topics<sec-advanced_topics-intro>` section holds references for developers and contributors to the Eradiate project.

About
-----

Eradiate's development is funded by a European Space Agency project funded by the European Commission's Copernicus programme. The design phase was funded by the MetEOC-3 project.

Eradiate's core development team consists of Yves Govaerts, Vincent Leroy, Yvan Nollet and Sebastian Schunke.

Eradiate inherits its core infrastructure from the `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_ renderer and the Eradiate team acknowledges all Mitsuba 2 contributors for their exceptional work.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   rst/getting_started/intro
   rst/getting_started/getting_code
   rst/getting_started/building

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   rst/tutorials/intro

.. toctree::
   :maxdepth: 1
   :caption: Advanced topics

   rst/advanced_topics/intro

.. toctree::
    :maxdepth: 1
    :caption: Miscellaneous

    rst/miscellaneous/bibliography

.. .. toctree::
..     :hidden:
..     :maxdepth: 1
..     :caption: Python interface

..     python_bindings/intro
..     python_bindings/rendering_scene
..     python_bindings/depth_integrator
..     python_bindings/direct_integrator
..     python_bindings/diffuse_bsdf

.. .. toctree::
..     :hidden:
..     :maxdepth: 1
..     :caption: Tutorials

..     tutorials/intro

.. .. toctree::
..     :hidden:
..     :maxdepth: 3
..     :caption: Advanced topics

..     advanced_topics/pythoninterface
..     advanced_topics/dev_guide
..     advanced_topics/plugin_docs
..     advanced_topics/api_reference

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
