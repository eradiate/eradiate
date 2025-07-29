.. _sec-user_guide-atmosphere_experiment:

One-dimensional experiment features
===================================

Eradiate ships with an :term:`experiment` dedicated to the simulation of
radiative transfer on one-dimensional geometries. While the computation is not
actually one-dimensional (Eradiate's kernel is a 3D Monte Carlo ray tracer),
scene setup ensures that the computed quantities are equivalent to what would
be obtained with a proper 1D code. This guide introduces the features,
configuration format and usage of this component.

Available features
------------------

Eradiate's one-dimensional experiment is implemented by the
:class:`.AtmosphereExperiment` class.
Instances are configured using the class's constructor. The
:term:`operational mode`, which defines how the spectral dimension of the
computation is handled, must be selected prior to instantiating the class.

The constructor only accepts keyword arguments, each of which can be passed an
object of an expected type or, alternatively and for relevant entries,
dictionaries. These dict-configurable entries all require a ``type`` parameter
which selects a programmatic component in Eradiate. In the following, the
``type`` value corresponding to each feature is specified in brackets.

In addition to the ``type`` parameter, an object configuration dictionary should
contain the parameters required to initialize the corresponding programmatic
element. Each allowed option is referenced, as well as the corresponding class
in the Eradiate codebase. Class documentation lists all their parameters and
allowed values.

Parameters can usually be omitted; in that case, they will be assigned
default values. Default values are documented in the reference documentation of
the :class:`.AtmosphereExperiment` class and other Eradiate classes.

Operational modes
^^^^^^^^^^^^^^^^^

The operational mode is selected using the :func:`eradiate.set_mode` function.
:class:`.AtmosphereExperiment` currently supports the monochromatic (``mono``) and
correlated-*k* distribution (``ckd``) modes.

Geometry [``geometry``]
^^^^^^^^^^^^^^^^^^^^^^^

Plane parallel [:class:`.PlaneParallelGeometry`, ``plane_parallel``]
    By default, the surface and atmosphere are assumed translationally invariant
    in the X and Y directions. This approximation provides satisfactory accuracy
    in many situations.

Spherical shell [:class:`.SphericalShellGeometry`, ``spherical_shell``]
    When this geometry configuration is used, the scene is built with a
    rotationally invariant symmetry. This configuration accounts for planetary
    curvature.

Illumination [``illumination``]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directional illumination [:class:`.DirectionalIllumination`, ``directional``]
    An infinitely distant emitter which emits light in a single direction
    (angular Dirac delta distribution of incoming radiance). This type of
    illumination is used to simulate incoming solar radiation.

Astronomical object [:class:`.AstroObjectIllumination`, ``astro_object``]
    An infinitely distant emitter that emits light in a conical section of the
    angular space. It aims at providing a more realistic representation of
    natural illuminants than the directional emitter.

In addition, this angular distribution can be associated a spectrum.
A variety of pre-defined Solar irradiance spectra are defined (see
:ref:`sec-data-solar_irradiance` for a complete list of shipped irradiance
spectra).

Measures [``measures``]
^^^^^^^^^^^^^^^^^^^^^^^

This experiment currently supports the computation of radiative quantities at
the top of the atmosphere. This parameter can be specified as a single measure,
or as a list of measures.

Distant radiancemeter [:class:`.MultiDistantMeasure`, ``distant``]
    This flexible measure records radiance exiting the scene. In practice, it
    outputs the top-of-atmosphere radiance under the set illumination
    conditions. The viewing directions for which radiance is computed can be
    controlled easily using various class method constructors.

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
        The TOA BRDF normalised by the BRDF of a non-absorbing diffuse
        (Lambertian) surface.

Distant fluxmeter [:class:`.DistantFluxMeasure`, ``distant_flux``]
    This measure records the flux leaving the scene (in W/m²/nm) over the entire
    hemisphere. It is mostly used to compute the scene albedo. The following
    quantities are available from the result dataset:

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
    altitude coordinate. The :class:`.HeterogeneousAtmosphere` class is
    configured by specifying a molecular component
    (:class:`.MolecularAtmosphere`), describing absorption and  scattering by
    atmospheric gases, and an arbitrary number of aerosol layers
    (:class:`.ParticleLayer`).

Surface [``surface``]
^^^^^^^^^^^^^^^^^^^^^

In this experiment, surfaces are smooth and their geometry is controlled by the
``geometry`` parameter. Only the surface's radiative properties can be selected.
The bidirectional scattering distribution function (BSDF) can be directly passed
as the ``surface`` parameter: Eradiate's internals will wrap them in an
appropriate shape.

Diffuse surface [:class:`.LambertianBSDF`, ``lambertian``]
    A diffuse or Lambertian surface reflects incoming radiation isotropically,
    regardless the incoming direction. This behaviour is modelled by the Lambert
    BRDF, parametrised by a reflectance parameter.

Rahman-Pinty-Verstraete (RPV) surface [:class:`.RPVBSDF`, ``rpv``]
    This reflection model features an anisotropic behaviour and is commonly
    used for land surface reflection modelling. Eradiate implements several
    variants of it with 3 or 4 parameters.

Ross thick-Li sparse (RTLS) surface [:class:`.RTLSBSDF`, ``rtls``]
    This reflection model is commonly used in remote sensing applications.

Hapke surface [:class:`.HapkeBSDF`]
    A reflection model specialized for bare soil.

Black surface [:class:`.BlackBSDF`, ``black``]
    The black surface absorbs all incoming radiation, irrespective of
    incident angle or wavelength.

Result output
-------------

When running an experiment with the :func:`eradiate.run` function, the computed
results are stored in the ``results`` attribute as a dictionary mapping measure
identifiers to a :class:`~xarray.Dataset` object. Each data set has one variable
for each computed physical quantity (*e.g.* spectral irradiance, leaving
radiance, BRDF and BRF for the ``distant`` measure). Results can then be easily
exported to files (*e.g.* NetCDF) and visualised using xarray's integrated
plotting features or external plotting components.
