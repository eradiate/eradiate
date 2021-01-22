.. _sec-getting_started-intro:

Getting started
===============

.. toctree::
   :maxdepth: 1

   quickstart
   getting_code
   building

What is Eradiate?
-----------------

Eradiate is a new-generation radiative transfer model for Earth observation
applications. It primarily targets calibration/validation applications and
focuses on delivering highly accurate results.

Eradiate is built around a computational kernel based on the Mitsuba 2 rendering
system. It provides abstractions to conveniently build scenes, manage kernel
runs and collect results.

What can I do with Eradiate
---------------------------

Perform monochromatic simulations.
    Eradiate simulates radiative transfer for a single wavelength between 280
    and 2400 nm.

Perform simulations on one-dimensional scenes.
    Eradiate supports simulations on 3D scenes carefully designed to produce
    results similar to what we would get with 1D geometries. These scenes
    consist of a flat surface underneath a cloud-free and aerosol-free
    atmosphere.

Perform simulations on three-dimensional scenes.
    Eradiate supports simulations on 3D scenes consisting of a vegetated ground
    patch with no atmosphere above it.

*More features are coming soon!*
