.. _sec-user_guide-basic_concepts:

Basic concepts and terminology
==============================

The basic workflow in Eradiate is split into three phases:

1. A scene suitable for the targeted application is generated.
2. Radiative transfer is computed on the created scene using a Monte Carlo
   ray tracing method.
3. Results yielded by the kernel are collected, post-processed and possibly
   plotted for visualisation.

In the following, Eradiate's basic components are introduced in the framework of
this workflow.

Kernel
------

Eradiate's low-level abstractions are handled by its radiometric *kernel*. This
component is a copy of the Mitsuba 2 rendering system with custom updates and it
implements the Monte Carlo integration framework Eradiate uses to compute
radiative transfer. The kernel takes as its input a
:term:`kernel scene description <kernel dictionary>` specified as a Python
dictionary and consisting of geometric shapes, radiative properties, but also
radiation sensors and emitters and a Monte Carlo integration algorithm. When
instructed to, the kernel loads the scene, performs the requested computation
and yields its raw results.

Eradiate's higher-level components are designed as abstractions of its kernel
and most users do not have to manipulate it by themselves.

Scene elements
--------------

*Scene elements* provide abstractions used to create input for the Eradiate
kernel. They let users easily create scenes of arbitrary complexity. Scene
elements combine kernel-level abstractions (shapes, spectra, BSDFs, media, etc.)
in a consistent way to describe physical items used to populate a scene. For
instance, a :class:`~eradiate.scenes.surface.Surface` scene element combines
a kernel shape and a BSDF to describe the surface in a one-dimensional scene.

All scene element components derive from the
:class:`~eradiate.scenes.core.SceneElement` abstract class and implement a
:meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method which generates
input data for the kernel.

Experiments
-----------

*Experiments* are the highest-level component of the operational
control chain in Eradiate. These are also the entry point for many users.
Applications parse user-generated configuration files and assemble scene
elements so that the runner can perform all necessary computations to yield
the data requested by the user.

Experiments are also the most specialised component of Eradiate. Users can
create dedicated applications for highly specific purposes if their use case
is not covered well by existing experiments. Eradiate ships experiments written
as specialised classes meant to be used in a script of an interactive console.
However, in a broader sense, an interactive Jupyter Lab session where a user
would assemble their scene and execute a computation can also be seen as an
experiment.

.. admonition:: Example

   The one-dimensional experiment
   (:class:`~eradiate.experiments.OneDimExperiment`)
   simulates radiative transfer in pseudo-one-dimensional geometries.
   Its configuration uses concepts with which Earth observation scientists
   should be familiar. The underlying machinery then breaks down this
   description into a set of kernel scene specifications, accounting for various
   constraints to ensure that the produced results are correct. The kernel scene
   dictionary is then passed to the Mitsuba 2 kernel to perform Monte Carlo ray
   tracing simulations, and the experiment object then collects the results and
   post-processes them.

   A top-of-atmosphere radiance-based measure (*e.g.* BRF) is specified to the
   :class:`~eradiate.experiments.OneDimExperiment` as a
   :class:`~eradiate.scenes.measure.MultiDistantMeasure`. This measure object is
   then translated into a ``mdistant`` sensor kernel plugin specification. The
   experiment object then runs the Monte Carlo ray tracing simulations and
   collects leaving radiance values. Radiance estimates are then further
   post-processed to compute reflectance quantities (BRF and BRDF).

Glossary
--------

.. glossary::

   Experiment
     A high-level description of a complete simulation including the scene,
     simulation parameters and post-processing routines.

   Film
     A kernel component which defines how samples collected by a sensor are
     stored in memory during kernel runs. This terminology originates from the
     graphics community and is a reference to cameras.

   Integrator
     A kernel component which implements a Monte Carlo ray tracing algorithm.
     Eradiate provides lightweight interface components to configure them.

   Kernel dictionary
     A dictionary describing the scene at the kernel level. Kernel dictionaries
     are created by combining kernel dict parts produced by the various scene
     elements in the scene and usually depend on contextual data.

   Measure
     A high-level interface to one or several :term:`sensors <sensor>`. Measures
     are associated to specific post-processing tasks managed by
     :class:`.Experiment` instances.

   Scene
     All kernel-level components required to perform a single radiative transfer
     simulation. The includes geometric shapes defining surfaces and volumes,
     radiative properties attached to them, emitters, sensors and an integrator.

   Sensor
     A kernel component which records radiance samples and stores them to a
     :term:`film`. Eradiate creates sensors from :term:`measures <measure>`.
