.. _sec-user_guide:

User guide
==========

The user guide introduces a range of topics explaining parts of Eradiate from a
user's perspective. Reading this material is not necessary to learn how to run
simulations with Eradiate—for this, it is advised to refer to the
:ref:`sec-tutorials`.

What is Eradiate?
-----------------

Eradiate is a modern radiative transfer model for Earth observation
applications. It primarily targets calibration/validation applications and
focuses on delivering highly accurate results.

Eradiate is built around a radiometric kernel based on the Mitsuba 3 rendering
system. It provides abstractions to conveniently build scenes, manage kernel
runs and collect results.

Features: What can I do with Eradiate?
--------------------------------------

Perform monochromatic and band simulations.
    Eradiate can simulate radiative transfer for a single wavelength or a
    spectral band (using the correlated *k*-distribution method) between in the
    solar reflective domain.

Perform simulations on one-dimensional scenes.
    Eradiate can simulate top-of-atmosphere and in-situ radiance and reflectance
    on 1D scenes consisting of a smooth surface underneath an atmosphere
    including gases and an arbitrary number of aerosol layers, and discretized
    into uniform, horizontally invariant layers. Under the hood, Eradiate
    performs these simulations using 3D scenes carefully designed to produce
    results similar to what one would get with 1D geometries.

    Eradiate supports plane-parallel and spherical-shell geometries.

Perform simulations on three-dimensional scenes.
    Eradiate can simulate top-of-canopy/atmosphere radiance and in-situ radiance
    and reflectance on 3D scenes consisting of a vegetated ground patch, with or
    without atmosphere above it.

.. seealso::

   For a full list of features, head to
   `our website <https://www.eradiate.eu/>`_.

Guides
------

.. toctree::
   :maxdepth: 1

   install
   basic_concepts
   config
   package_structure
   atmosphere_experiment
   dem_experiment
   canopy_scene_loader
   spectral_discretization
   conventions
   unit_guide_user
