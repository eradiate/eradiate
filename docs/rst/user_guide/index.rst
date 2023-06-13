.. _sec-user_guide:

User guides
===========

.. toctree::
   :maxdepth: 1

   install
   basic_concepts
   conventions
   package_structure
   onedim_experiment
   spectral_discretization
   unit_guide_user
   data/index

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
    spectral band (using the correlated *k*-distribution method) between 280
    and 2400 nm.

Perform simulations on one-dimensional scenes.
    Eradiate can simulate top-of-atmosphere radiance and reflectance on 1D
    scenes with plane parallel geometry.
    These scenes consist of a smooth surface underneath an atmosphere including
    gases and an arbitrary number of aerosol layers. Under the hood, Eradiate
    performs these simulations using 3D scenes carefully designed to produce
    results similar to what one would get with 1D geometries.

    Eradiate has experimental support for spherical shell geometries (validation
    is in progress).

Perform simulations on three-dimensional scenes.
    Eradiate can simulate top-of-canopy/atmosphere radiance on 3D scenes
    consisting of a vegetated ground patch with or without atmosphere above it
    (plane parallel geometry only).

.. seealso::

   For a full list of features, head to
   `our website <https://www.eradiate.eu/>`_.