.. _sec-user_guide-onedim_solver_app:

One-dimensional solver application
==================================

Eradiate ships with a solver application dedicated to the simulation of
radiative transfer on one-dimensional geometries. While the computation is not
actually one-dimensional (Eradiate's kernel is a 3D Monte Carlo ray tracer),
scene setup ensures that the computed quantities are equivalent to what would
be obtained with a proper 1D code.

As mentioned in the :ref:`sphx_glr_examples_generated_tutorials_01_quick_overview.py`
tutorial, the one-dimensional solver application can be used either through its
command-line interface ``ertonedim`` or through its Python API (see
:class:`.OneDimSolverApp`). This guide goes in more detail on the
features, configuration format and usage of this application.

.. admonition:: Tutorials

   * Basic usage ⇒ :ref:`sphx_glr_examples_generated_tutorials_solver_onedim_01_solver_onedim.py`
   * Simulations on heterogeneous atmospheres ⇒ :ref:`sphx_glr_examples_generated_tutorials_solver_onedim_02_onedim_sim_hetero_atm.py`

Available features
------------------

Eradiate's one-dimensional solver is implemented by the :class:`.OneDimSolverApp`
class. The different ways of configuring this application are presented in
:ref:`sphx_glr_examples_generated_tutorials_solver_onedim_01_solver_onedim.py`.

The application can be configured by using its Python constructor. In this case,
the operational mode must be selected prior to creating the
:class:`.OneDimSolverApp` instance.
The entire application can also be configured using a dictionary divided into
sections presented and detailed below. Unless specified, each section is also a
dictionary and should take a ``type`` parameter which selects a programmatic
component in Eradiate. In the following, the ``type`` value corresponding to
each feature is specified in brackets.

In addition to the ``type`` parameter, a section should contain the parameters
required to initialise the corresponding programmatic element. Each allowed
option is referenced, as well as the corresponding class in the Eradiate
codebase. Class documentation details all their parameters and allowed values.

Sections and parameters can be omitted; in that case, they will be assigned
default values. Default section values are documented in the reference
documentation of the :class:`.OneDimSolverApp` class and the other Eradiate
classes.

When parameters are mentioned as "unit-enabled" in the reference documentation,
it means that they can be assigned units in a field bearing the same name with
the suffix ``_units``. See the
:ref:`Unit guide <sec-user_guide-unit_guide_user-field_unit_documentation>`
for more detail. See below for practical usage of this feature.

The command-line interface (CLI) to :class:`.OneDimSolverApp` uses a dictionary
to configure the app, defined in a YAML file.

Operational mode [``mode``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``mode`` section is only applicable if configuring :class:`.OneDimSolverApp`
with a dictionary. It configures Eradiate's operational mode. It notably configures
the computational kernel. The ``type`` parameter must be a valid mode identifier.

If you are using the :class:`.OneDimSolverApp` constructor, you then have to
select the operational mode by yourself using :func:`eradiate.set_mode`.

Monochromatic mode [``mono``, ``mono_double``]
    This application currently supports the ``mono`` mode, which
    performs monochromatic simulations. In this mode, only one wavelength is
    transported per ray traced by the Monte Carlo engine. The ``mono`` mode is
    wavelength-aware and has a single ``wavelength`` parameter. A double-precision
    variant ``mono_double`` can also be selected.

Illumination [``illumination``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The one-dimensional solver application currently supports only one illumination
type.

Directional illumination [:class:`.DirectionalIllumination`, ``directional``]
    An infinitely distant emitter emits light in a single direction (angular
    Dirac delta distribution of incoming radiance). This type of illumination is
    used to simulate incoming Solar radiation.

In addition, this angular distributions can be associated a spectrum.
A variety of pre-defined Solar irradiance spectra are defined (see
:mod:`~eradiate.data.solar_irradiance_spectra` for a complete list).
Custom-defined spectrum definitions are also supported.

Measure [``measure``]
^^^^^^^^^^^^^^^^^^^^^

This application currently supports the computation of radiative quantities at
the top of the atmosphere.

Distant measure [:class:`.DistantMeasure`]
    This flexible measure places a sensor at the top of the atmosphere. It
    therefore records the radiance leaving the scene. It can be set to record
    radiance over the entire hemisphere or in a plane. The recorded outgoing
    radiance is then post-processed as detailed hereafter. In the next
    paragraph, quantities are associated to the name of their corresponding
    field in post-processed results.

    TOA outgoing radiance [``lo``]
        This is the radiance reflected by the entire scene (surface and atmosphere),
        since the scene only contains infinitely distant illumination.

    TOA bidirectional reflectance distribution function (TOA BRDF) [``brdf``]
        The TOA leaving radiance can be post-processed together with scene
        illumination parameters to compute the TOA BRDF.

    TOA bidirectional reflectance factor (TOA BRF) [``brf``]
        This is simply the TOA BRDF normalised by the BRDF of a non-absorbing
        diffuse (Lambertian) surface.

Atmosphere [``atmosphere``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An atmosphere can be optionally added to the scene. Currently, two types of
atmosphere are supported.

Homogeneous atmosphere [:class:`.HomogeneousAtmosphere`, ``homogeneous``]
    The atmosphere has spatially invariant radiative properties.

Heterogeneous atmosphere [:class:`.HeterogeneousAtmosphere`, ``heterogeneous``]
    The atmosphere has spatially varying radiative properties along the
    altitude coordinate. Absorption and Rayleigh scattering are currently
    supported.

Surface [``surface``]
^^^^^^^^^^^^^^^^^^^^^

In this application, surfaces are plane and their geometry cannot be adjusted.
Only the surface's radiative properties can be selected.

Diffuse surface [:class:`.LambertianSurface`, ``lambertian``]
    A diffuse or Lambertian surface reflects incoming radiation isotropically,
    regardless the incoming direction. This behaviour is modelled by the Lambert
    BRDF, parametrised by a reflectance parameter.

Rahman-Pinty-Verstraete (RPV) surface [:class:`.RPVSurface`, ``rpv``]
    This reflection model features an anisotropic behaviour and is commonly
    used for land surface reflection modelling. Eradiate implements several
    variants of it with 3 or 4 parameters.

Black surface [:class:`.BlackSurface`, ``black``]
    The black surface absorbs all incoming radiation, irrespective of
    incident angle or wavelength.

Result output
-------------

Data output depends on the way the application is accessed:

* The CLI outputs results to netCDF files whose naming pattern is controlled by
  a positional argument ``fname_results``, used as a file name prefix for all
  output data sets. One netCDF file is produced for each measure.
* When using the :class:`.OneDimSolverApp` class directly, the
  :meth:`~.OneDimSolverApp.run()` method stores the computed results in the
  ``results`` attribute as a dictionary mapping measure identifiers to a
  :class:`xarray.Dataset` object. Each data set has one variable for each
  computed physical quantity (*e.g.* spectral irradiance, leaving radiance, BRDF
  and BRF for the ``distant`` measure).

Visualisation
-------------

Visualisation also depends on how the application is accessed:

* The CLI outputs a series of default plots for each measure. The plot file
  naming pattern is controlled by a positional argument ``fname_plots``, used as
  a file name prefix for all plot files.
* When using the :class:`.OneDimSolverApp` class directly, the
  :meth:`~.OneDimSolverApp.run()` method does not produce plots. Plotting is left
  to the user.
