.. _sec-user_guide-package_structure:

Package structure overview
==========================

This section documents the general principles underlying the organisation of
Eradiate's Python packages. Eradiate's code is organised into five submodules:

:mod:`eradiate.kernel`
    This submodule provides an interface to Eradiate's computational kernel.

:mod:`eradiate.scenes`
    This library contains scene generation components.

:mod:`eradiate.solvers`
    Components in this submodule glue the previous together, coordinate kernel
    runs, implement post-processing and visualisation for specific use-cases.

:mod:`eradiate.util`
    This submodule contains additional utility components, not specific to any
    of the others.

:mod:`eradiate.data`
    Data shipped with Eradiate can be accessed using a generic interface defined
    in this submodule.

In the following, we will describe the organisation of the :mod:`eradiate.scenes`
module.

The high-level scene description library [:mod:`eradiate.scenes`]
-----------------------------------------------------------------

The scene generation API uses primary abstractions defined in the
:mod:`eradiate.scenes.core` module. this module defines the
:class:`~eradiate.scenes.core.SceneElement` base class as well as the
:class:`eradiate.scenes.core.KernelDict` class, which serves as the main
interface with the kernel.

The :mod:`eradiate.scenes` module also has a number of submodules classifying
components based on the Earth system science nomenclature:

:mod:`eradiate.scenes.atmosphere`
    This submodule contains scene elements used to add an atmosphere to the
    scene.

:mod:`eradiate.scenes.biosphere` (upcoming)
    This submodule will contain scene elements used to add vegetation
    to the scene.

:mod:`eradiate.scenes.cryosphere` (upcoming)
    This submodule will contain scene elements used to add cryosphere elements
    to the scene.

:mod:`eradiate.scenes.hydrosphere` (upcoming)
    This  submodule will contain scene elements used to add water surfaces and
    bodies to the scene.

:mod:`eradiate.scenes.lithosphere` (upcoming)
    This submodule will contain scene elements used to add mineral surfaces
    to the scene.

In addition, transverse components are defined in the following submodules:

:mod:`eradiate.scenes.surface`
    This submodule contains scene elements used to add a surface to the scene.

:mod:`eradiate.scenes.illumination`
    This module defines scene elements used to illuminate the scene, _i.e._
    light sources.

:mod:`eradiate.scenes.measure`
    This module defines scene elements used to compute radiative quantities.
