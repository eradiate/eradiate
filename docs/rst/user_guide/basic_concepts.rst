.. _sec-user_guide-basic_concepts:

Basic concepts and terminology
==============================

The basic workflow in Eradiate is split into three phases:

1. Prepare a scene suitable for the targeted application.
2. Compute radiative transfer on the created scene using a Monte Carlo ray
   tracing method.
3. Collect, post-process and plot results yielded by the radiometric kernel.

In the following, Eradiate's basic components are introduced in the framework of
this workflow.

Kernel
------

Eradiate's low-level abstractions are handled by its *radiometric kernel*. This
component consists of the Mitsuba 3 rendering system with custom extensions and
it implements the Monte Carlo integration framework Eradiate uses to compute
radiative transfer. The kernel takes as its input a
:term:`kernel scene description <kernel dictionary>` specified as a Python
dictionary and defining low-level representations of geometric shapes, radiative
properties, but also radiation sensors and emitters and a Monte Carlo
integration algorithm. When instructed to, the kernel loads the scene, performs
the requested computation and yields its raw results.

Eradiate's higher-level components are abstractions on top of the radiometric
kernel and most users do not have to manipulate it directly.

.. _sec-user_guide-basic_concepts-scene_elements:

Scene elements
--------------

*Scene elements* provide abstractions used to create input for the radiometric
kernel. They let users easily create scenes which would otherwise require
careful assembly. Scene elements combine kernel-level abstractions (shapes,
spectra, BSDFs, media, etc.) in a consistent way to describe physical items
used to populate a scene. For instance, a
:class:`~eradiate.scenes.surface.Surface` scene element combines a kernel shape
and a BSDF to describe the surface in a one-dimensional scene.

All scene element components derive from the
:class:`~eradiate.scenes.core.SceneElement` abstract class and implement
methods which generates input data for the kernel.

Experiments
-----------

*Experiments* are the highest-level component of the operational
control chain in Eradiate. These are also the entry point for many users.
Experiments interpret user-defined configuration and assemble scene elements so
that the radiometric kernel can perform all necessary computations and yield the
data requested by the user.

Experiments are also the most specialized component of Eradiate. The long-term
goal is to allow users to create dedicated applications for highly specific
purposes if their use case is not covered well by existing experiments. Eradiate
ships experiments written as specialized classes meant to be used in a script or
an interactive console. However, in a broader sense, an interactive Jupyter Lab
session where a user would assemble their scene and execute a computation can
also be seen as an experiment.

.. admonition:: Example
   :class: tip

   The one-dimensional experiment
   (:class:`~eradiate.experiments.AtmosphereExperiment`)
   simulates radiative transfer in pseudo-one-dimensional geometries.
   Its configuration uses concepts with which Earth observation scientists
   should be familiar. The underlying machinery then breaks down this
   description into a set of kernel scene specifications, accounting for various
   constraints to ensure that the produced results are correct. The kernel scene
   dictionary is then passed to the radiometric kernel to perform Monte Carlo
   ray tracing simulations, and the experiment object then collects the results
   and post-processes them.

   A top-of-atmosphere radiance-based measure (*e.g.* BRF) is specified to the
   :class:`~eradiate.experiments.AtmosphereExperiment` as a
   :class:`~eradiate.scenes.measure.MultiDistantMeasure`. This measure object is
   then translated into a :ref:`mdistant <plugin-sensor-mdistant>` kernel sensor
   plugin specification. The experiment object then runs the Monte Carlo ray
   tracing simulations and collects leaving radiance values. Radiance estimates
   are then further post-processed to compute reflectance quantities (BRF and
   BRDF).

Glossary
--------

.. glossary::

   Data store
     A location from where Eradiate fetches its shipped data. Data stores can
     be offline (directories) or online.

   Experiment
     A high-level description of a complete simulation including the scene,
     simulation parameters and post-processing routines.

   Film
     A kernel-level component which defines how samples collected by a sensor
     are stored in memory during kernel runs. This terminology originates from
     the graphics community and is a reference to cameras.

   Integrator
     A kernel-level component which implements a Monte Carlo ray tracing
     algorithm. Eradiate provides lightweight interface components to configure
     them.

   Measure
     A high-level interface to one or several :term:`sensors <sensor>`. Measures
     are associated to specific post-processing tasks managed by
     :class:`.Experiment` instances.

   Operational mode
     A global configuration item for Eradiate defining how the spectral
     dimension of the radiometric computation is handled. Currently, Eradiate
     supports the line-by-line and correlated-*k* modes.

   Scene
     All kernel-level components required to perform a single radiative transfer
     simulation. This includes geometric shapes defining surfaces and volumes,
     radiative properties attached to them, emitters, sensors and an integrator.

   Sensor
     A kernel component which records radiance samples and stores them to a
     :term:`film`. Eradiate creates sensors from :term:`measures <measure>`.

.. seealso::

   The :ref:`sec-developer_guides-radiometric_kernel_interface` section contains
   an additional specific glossary.
