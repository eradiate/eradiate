.. _sec-user_guide-onedim_experiment:

One-dimensional experiment features
===================================

Eradiate ships with an :term:`experiment` dedicated to the simulation of
radiative transfer on one-dimensional geometries. While the computation is not
actually one-dimensional (Eradiate's kernel is a 3D Monte Carlo ray tracer),
scene setup ensures that the computed quantities are equivalent to what would
be obtained with a proper 1D code.

This guide introduces the features, configuration format and usage of Eradiate's
one-dimensional experiment.

.. .. admonition:: Tutorials

..    * Basic usage ⇒ :ref:`sphx_glr_examples_generated_tutorials_experiment_onedim_01_experiment_onedim.py`
..    * Simulations on heterogeneous atmospheres ⇒ :ref:`sphx_glr_examples_generated_tutorials_experiment_onedim_02_onedim_sim_hetero_atm.py`

Available features
------------------

Eradiate's one-dimensional experiment is implemented by the
:class:`.OneDimExperiment` class.
Instances are configured by using the class's constructor. As usual,
the operational mode must be selected prior to instantiating the class.

The constructor only accepts keyword arguments, each of which can be passed an
object of an expected type or, alternatively and for relevant entries,
dictionaries. These dict-configurable entries all require a ``type`` parameter
which selects a programmatic component in Eradiate. In the following, the
``type`` value corresponding to each feature is specified in brackets.

In addition to the ``type`` parameter, an object configuration dictionary should
contain the parameters required to initialise the corresponding programmatic
element. Each allowed option is referenced, as well as the corresponding class
in the Eradiate codebase. Class documentation lists all their parameters and
allowed values.

Parameters can usually be omitted; in that case, they will be assigned
default values. Default values are documented in the reference documentation of
the :class:`.OneDimExperiment` class and other Eradiate classes.

Operational modes
^^^^^^^^^^^^^^^^^

The operational mode is selected using the :func:`eradiate.set_mode` function.
:class:`.OneDimExperiment` currently supports the monochromatic (``mono``) and
correlated-`k` distribution (``ckd``) modes.`

Illumination [``illumination``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The one-dimensional solver application currently supports only one illumination
type.

Directional illumination [:class:`.DirectionalIllumination`, ``directional``]
    An infinitely distant emitter emits light in a single direction (angular
    Dirac delta distribution of incoming radiance). This type of illumination is
    used to simulate incoming Solar radiation.

In addition, this angular distribution can be associated a spectrum.
A variety of pre-defined Solar irradiance spectra are defined (see
:mod:`~eradiate.data.solar_irradiance_spectra` for a complete list of shipped
irradiance spectrum datasets).

Measure [``measure``]
^^^^^^^^^^^^^^^^^^^^^

This application currently supports the computation of radiative quantities at
the top of the atmosphere.

Distant radiancemeter [:class:`.MultiDistantMeasure`, ``distant``]
    This flexible measure records radiance exiting the scene. In practice, it
    outputs the top-of-atmosphere radiance under the set illumination
    conditions. The viewing directions for which radiance is computed can be
    controlled easily using the :meth:`.MultiDistantMeasure.from_viewing_angles`
    constructor.

    When this measure is used, a number of derived quantities are
    computed. In the next paragraph, quantities available after post-processing
    are associated to the name of their corresponding field in the results
    dataset.

    TOA outgoing radiance [``radiance``]
        This is the radiance reflected by the entire scene (surface and
        atmosphere), since the scene only contains infinitely distant
        illumination.

    TOA bidirectional reflectance distribution function (TOA BRDF) [``brdf``]
        The TOA leaving radiance is post-processed together with scene
        illumination parameters to compute the TOA BRDF.

    TOA bidirectional reflectance factor (TOA BRF) [``brf``]
        The TOA BRDF normalised by the BRDF of a non-absorbing
        diffuse (Lambertian) surface.

Distant fluxmeter [:class:`.DistantFlux`, ``distant_flux``]
    This measure records the flux leaving the scene (in W/m²/nm) over the entire
    hemisphere. It is mostly used to compute the scene albedo. The following
    quantities are available from the results dataset:

    Radiosity [``radiosity``]
        The flux leaving the scene in W/m²/nm.

    Albedo [``albedo``]
        The total scene albedo.

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

The :meth:`.OneDimExperiment.run` method stores the computed results in the
``results`` attribute as a dictionary mapping measure identifiers to a
:class:`~xarray.Dataset` object. Each data set has one variable for each
computed physical quantity (*e.g.* spectral irradiance, leaving radiance, BRDF
and BRF for the ``distant`` measure). Results can then be easily exported to
files (*e.g.* NetCDF) and visualised using xarray's integrated plotting
features or external plotting libraries.
