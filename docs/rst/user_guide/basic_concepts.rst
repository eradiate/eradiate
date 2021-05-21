.. _sec-user_guide-basic_concepts:

Basic concepts and terminology
==============================

The basic workflow in Eradiate is split into three phases:

1. A scene suitable for the targeted application is generated.
2. Radiative transfer is computed on the created scene using a Monte Carlo
   ray tracing method.
3. Results yielded by the kernel are collected, post-processed and possibly
   plotted for visualisation.

In the following, the fundamental components used in Eradiate are introduced
in the framework of this workflow.

Kernel
------

Eradiate's low-level abstractions are handled by its computational kernel. This
component is a modified copy of the Mitsuba 2 rendering system and it implements
the Monte Carlo integration framework Eradiate uses to compute radiative
transfer. The kernel takes as its input a *scene*, most often specified as a
Python dictionary and consisting of geometric shapes, radiative properties,
but also radiation sensors and emitters and a Monte Carlo integration algorithm.
When instructed to, the kernel loads the scene, performs the requested
computation and yields the results.

Eradiate's higher-level components are designed as abstractions of its kernel
and most users do not have to manipulate it by themselves.

Scene elements
--------------

Scene elements provide abstractions used to create input for the Eradiate
kernel. They let users easily create scenes of arbitrary complexity. Scene
elements combine kernel-level abstractions (shapes, spectra, BSDFs, media, etc.)
in a consistent way to describe physical items used to populate a scene. For
instance, a :class:`~eradiate.scenes.surface.Surface` scene element combines
a kernel shape and a BSDF to describe the surface in a one-dimensional scene.

All scene element components derive from the :class:`~eradiate.scenes.core.SceneElement`
abstract class and implement a :meth:`~eradiate.scenes.core.SceneElement.kernel_dict`
method which generates input data for the kernel.

..

   Runners [COMMENTED: RETIRED FOR NOW]
   ------------------------------------

   Runner components are responsible for managing kernel runs and make the
   interface between the scene generation API and the kernel. They take as their
   input the output produced by scene element and pass it to the kernel. They then
   run and monitor the radiative transfer computation.

   They also keep track of kernel scene features and associate them with Eradiate's
   high-level scene abstractions. For instance, the runner is responsible for
   ensuring that all requested high-level measurements are computed and their
   results organised in such a way that high-level post-processing is possible.

   Runners also supervise kernel runs to handle the presence of multiple sensors
   (sometimes required to compute complex measurements) or chunk large spectral
   intervals into smaller bits.

Applications
------------

Application components are the highest-level component of the operational
control chain in Eradiate. These are also the entry point for many users.
Applications parse user-generated configuration files and assemble scene
elements so that the runner can perform all necessary computations to yield
the data requested by the user.

Applications are also the most specialised component of Eradiate. Users can
create dedicated applications for highly specific purposes if their use case
is not covered well by existing applications. Eradiate ships applications
written as scripts with a command-line interface; however, in a broader sense,
an interactive Jupyter Lab session where a user would assemble their scene and
execute a computation can also be seen as an application.

Application design is flexible and can leverage all the features of the scene
generation API. The scene element interface is notably designed in a way such
that it is very simple to write an application which will allow direct
specification of scene elements in a configuration file.

.. admonition:: Example

   The one-dimensional solver application (:class:`~eradiate.solvers.onedim.OneDimSolverApp`)
   simulates radiative transfer in pseudo-one-dimensional geometries.
   Its configuration uses concepts with which Earth observation scientists
   should be familiar. The underlying machinery then breaks down this
   description into a set of kernel scene specification, accounting for various
   constraints to ensure that the produced results are correct. A runner
   component uses these kernel scene specification to perform Monte Carlo ray
   tracing simulations, and the application then collects the results and
   post-processes them.

   A top-of-atmosphere radiance-based measure (*e.g.* BRF) is specified to the
   :class:`~eradiate.solvers.onedim.OneDimSolverApp` as a
   :class:`~eradiate.scenes.measure.DistantMeasure`. This measure object is
   then translated into a ``distant`` sensor kernel plugin specification. The
   application then runs the Monte Carlo ray tracing simulations and collects
   leaving radiance values. Radiance estimates are then further post-processed
   to compute reflectance quantities (BRF and BRDF).

Other terminology
-----------------

Integrator
  A kernel component which implements a Monte Carlo ray tracing algorithm.
  Eradiate provides lightweight interface components to configure them.

Sensor
  A kernel component which records radiance samples and stores them to a film.

Film
  A kernel component which defines how samples collected by a sensor are stored.

Measure
  A high-level interface to one or several sensors. Measures can perform
  post-processing tasks requiring the assembly of multiple sensor results. They
  also output their results as high-level data structures (*e.g.* xarray
  labelled arrays).
