``eradiate.scenes``
===================

.. automodule:: eradiate.scenes

Quick access
------------

.. grid:: 1 2 auto auto
    :gutter: 3

    .. grid-item-card:: :iconify:`material-symbols:settings height=1.5em` Core
        :link: module-eradiate.scenes.core
        :link-type: ref

        ``eradiate.scenes.core``

    .. grid-item-card:: :iconify:`material-symbols:language height=1.5em` Geometry
        :link: module-eradiate.scenes.geometry
        :link-type: ref

        ``eradiate.scenes.geometry``

    .. grid-item-card:: :iconify:`material-symbols:cloud height=1.5em` Atmosphere
        :link: module-eradiate.scenes.atmosphere
        :link-type: ref

        ``eradiate.scenes.atmosphere``

    .. grid-item-card:: :iconify:`material-symbols:forest height=1.5em` Biosphere
        :link: module-eradiate.scenes.biosphere
        :link-type: ref

        ``eradiate.scenes.biosphere``

    .. grid-item-card:: :iconify:`material-symbols:landscape height=1.5em` Surface
        :link: module-eradiate.scenes.surface
        :link-type: ref

        ``eradiate.scenes.surface``

    .. grid-item-card:: :iconify:`material-symbols:trending-up height=1.5em` BSDFs
        :link: module-eradiate.scenes.bsdfs
        :link-type: ref

        ``eradiate.scenes.bsdfs``


    .. grid-item-card:: :iconify:`material-symbols:dashboard height=1.5em` Shapes
        :link: module-eradiate.scenes.shapes
        :link-type: ref

        ``eradiate.scenes.shapes``

    .. grid-item-card:: :iconify:`material-symbols:sunny height=1.5em` Illumination
        :link: module-eradiate.scenes.illumination
        :link-type: ref

        ``eradiate.scenes.illumination``

    .. grid-item-card:: :iconify:`material-symbols:videocam height=1.5em` Measure
        :link: module-eradiate.scenes.measure
        :link-type: ref

        ``eradiate.scenes.measure``

    .. grid-item-card:: :iconify:`material-symbols:trending-down height=1.5em` Phase functions
        :link: module-eradiate.scenes.phase
        :link-type: ref

        ``eradiate.scenes.phase``

    .. grid-item-card:: :iconify:`material-symbols:dns height=1.5em` Integrators
        :link: module-eradiate.scenes.integrators
        :link-type: ref

        ``eradiate.scenes.integrators``

    .. grid-item-card:: :iconify:`material-symbols:looks height=1.5em` Spectra
        :link: module-eradiate.scenes.spectra
        :link-type: ref

        ``eradiate.scenes.spectra``

.. _module-eradiate.scenes.core:

``eradiate.scenes.core``
------------------------

.. automodule:: eradiate.scenes.core

.. py:currentmodule:: eradiate.scenes.core

**Scene element traversal**

.. autosummary::
   :toctree: generated/autosummary/

   traverse
   SceneTraversal

**Basic scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   Ref
   Scene

**Scene element interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   SceneElement
   NodeSceneElement
   InstanceSceneElement
   CompositeSceneElement

**Miscellaneous**

.. autosummary::
   :toctree: generated/autosummary/

   BoundingBox

.. _module-eradiate.scenes.geometry:

``eradiate.scenes.geometry``
----------------------------

.. automodule:: eradiate.scenes.geometry

.. py:currentmodule:: eradiate.scenes.geometry

.. autosummary::
   :toctree: generated/autosummary/

   SceneGeometry
   PlaneParallelGeometry
   SphericalShellGeometry

.. _module-eradiate.scenes.atmosphere:

``eradiate.scenes.atmosphere``
------------------------------

.. automodule:: eradiate.scenes.atmosphere

.. py:currentmodule:: eradiate.scenes.atmosphere

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Atmosphere
   AbstractHeterogeneousAtmosphere
   ParticleDistribution

**Factories**

* :data:`atmosphere_factory`
* :data:`particle_distribution_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   HomogeneousAtmosphere
   HeterogeneousAtmosphere
   MolecularAtmosphere
   ParticleLayer

**Particle distributions**

.. autosummary::
   :toctree: generated/autosummary

   ArrayParticleDistribution
   ExponentialParticleDistribution
   InterpolatorParticleDistribution
   GaussianParticleDistribution
   UniformParticleDistribution

**Utility functions**

.. autosummary::
   :toctree: generated/autosummary

   eval_transmittance_mono
   eval_transmittance_ckd

.. _module-eradiate.scenes.biosphere:

``eradiate.scenes.biosphere``
-----------------------------

.. automodule:: eradiate.scenes.biosphere

.. py:currentmodule:: eradiate.scenes.biosphere

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Canopy
   CanopyElement

**Factories**

* :data:`biosphere_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   AbstractTree
   DiscreteCanopy
   InstancedCanopyElement
   LeafCloud
   MeshTree

**Mesh-base tree components**

.. autosummary::
   :toctree: generated/autosummary/

   MeshTreeElement

**Parameters for LeafCloud generators**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _leaf_cloud.ConeLeafCloudParams
      _leaf_cloud.CuboidLeafCloudParams
      _leaf_cloud.CylinderLeafCloudParams
      _leaf_cloud.EllipsoidLeafCloudParams
      _leaf_cloud.SphereLeafCloudParams

**Canopy loader**

.. autosummary::
   :toctree: generated/autosummary/

   load_scenario
   load_rami_scenario
   RAMIActualCanopies
   RAMIHeterogeneousAbstractCanopies
   RAMIHomogeneousAbstractCanopies
   RAMIScenarioVariant

.. _module-eradiate.scenes.surface:

``eradiate.scenes.surface``
---------------------------

.. automodule:: eradiate.scenes.surface

.. py:currentmodule:: eradiate.scenes.surface

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Surface

**Factories**

* :data:`surface_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   BasicSurface
   CentralPatchSurface
   DEMSurface

**Helpers**

.. autosummary::
   :toctree: generated/autosummary/

   mesh_from_dem
   triangulate_grid

.. _module-eradiate.scenes.bsdfs:

``eradiate.scenes.bsdfs``
-------------------------

.. automodule:: eradiate.scenes.bsdfs

.. py:currentmodule:: eradiate.scenes.bsdfs

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   BSDF

**Factories**

* :data:`bsdf_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   BlackBSDF
   CheckerboardBSDF
   HapkeBSDF
   LambertianBSDF
   MaignanBSDF
   MQDiffuseBSDF
   OceanGraspBSDF
   OceanLegacyBSDF
   OceanMishchenkoBSDF
   OpacityMaskBSDF
   RPVBSDF
   RTLSBSDF

.. _module-eradiate.scenes.shapes:

``eradiate.scenes.shapes``
--------------------------

.. automodule:: eradiate.scenes.shapes

.. py:currentmodule:: eradiate.scenes.shapes

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Shape
   ShapeNode
   ShapeInstance

**Factories**

* :data:`shape_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   BufferMeshShape
   CuboidShape
   FileMeshShape
   RectangleShape
   SphereShape

.. _module-eradiate.scenes.illumination:

``eradiate.scenes.illumination``
--------------------------------

.. automodule:: eradiate.scenes.illumination

.. py:currentmodule:: eradiate.scenes.illumination

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Illumination
   AbstractDirectionalIllumination

**Factories**

* :data:`illumination_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   DirectionalIllumination
   AstroObjectIllumination
   ConstantIllumination
   SpotIllumination

.. _module-eradiate.scenes.measure:

``eradiate.scenes.measure``
---------------------------

.. automodule:: eradiate.scenes.measure

.. py:currentmodule:: eradiate.scenes.measure

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Measure
   Target

**Factories**

* :data:`measure_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   DistantMeasure
   MultiDistantMeasure
   MultiPixelDistantMeasure
   DistantFluxMeasure
   HemisphericalDistantMeasure
   RadiancemeterMeasure
   MultiRadiancemeterMeasure
   PerspectiveCameraMeasure

**Distant measure target definition**

.. autosummary::
  :toctree: generated/autosummary/

  TargetPoint
  TargetRectangle

**Viewing direction layouts**

*Used as input to the* :class:`.MultiDistantMeasure`\ ``.layout`` *field.*

.. autosummary::
   :toctree: generated/autosummary/

   Layout
   AngleLayout
   AzimuthRingLayout
   DirectionLayout
   GridLayout
   HemispherePlaneLayout

.. _module-eradiate.scenes.phase:

``eradiate.scenes.phase``
-------------------------

.. automodule:: eradiate.scenes.phase

.. py:currentmodule:: eradiate.scenes.phase

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   PhaseFunction

**Factories**

* :data:`phase_function_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   IsotropicPhaseFunction
   RayleighPhaseFunction
   HenyeyGreensteinPhaseFunction
   BlendPhaseFunction
   TabulatedPhaseFunction

.. _module-eradiate.scenes.integrators:

``eradiate.scenes.integrators``
-------------------------------

.. automodule:: eradiate.scenes.integrators

.. py:currentmodule:: eradiate.scenes.integrators

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Integrator

**Factories**

* :data:`integrator_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   PathIntegrator
   PiecewiseVolPathIntegrator
   VolPathIntegrator
   VolPathMISIntegrator

.. _module-eradiate.scenes.spectra:

``eradiate.scenes.spectra``
---------------------------

.. automodule:: eradiate.scenes.spectra

.. py:currentmodule:: eradiate.scenes.spectra

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Spectrum

**Factories**

* :data:`spectrum_factory`

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _core.SpectrumFactory

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   UniformSpectrum
   InterpolatedSpectrum
   SolarIrradianceSpectrum
   AirScatteringCoefficientSpectrum
