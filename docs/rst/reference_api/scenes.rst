``eradiate.scenes``
===================

.. automodule:: eradiate.scenes

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

``eradiate.scenes.geometry``
----------------------------

.. automodule:: eradiate.scenes.geometry

.. py:currentmodule:: eradiate.scenes.geometry

.. autosummary::
   :toctree: generated/autosummary/

   SceneGeometry
   PlaneParallelGeometry
   SphericalShellGeometry

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

**Full canopies**

.. autosummary::
   :toctree: generated/autosummary/

   wellington_citrus_orchard

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
   LambertianBSDF
   MQDiffuseBSDF
   OpacityMaskBSDF
   RPVBSDF

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

``eradiate.scenes.illumination``
--------------------------------

.. automodule:: eradiate.scenes.illumination

.. py:currentmodule:: eradiate.scenes.illumination

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Illumination

**Factories**

* :data:`illumination_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   DirectionalIllumination
   AstroObjectIllumination
   ConstantIllumination
   SpotIllumination

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

   MultiDistantMeasure
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
   VolPathIntegrator
   VolPathMISIntegrator

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
   MultiDeltaSpectrum
   SolarIrradianceSpectrum
   AirScatteringCoefficientSpectrum
