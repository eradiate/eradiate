.. _sec-reference-scenes:

Scene generation [eradiate.scenes]
==================================

.. _sec-reference-scenes-core:

Core [eradiate.scenes.core]
---------------------------
.. currentmodule:: eradiate.scenes.core

.. autosummary::
   :toctree: generated/autosummary/

   SceneElement
   KernelDict
   BoundingBox

.. _sec-reference-scenes-atmosphere:

Atmosphere [eradiate.scenes.atmosphere]
---------------------------------------
.. currentmodule:: eradiate.scenes.atmosphere

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Atmosphere
   AbstractHeterogeneousAtmosphere

**Factories**

* :data:`atmosphere_factory`
* :data:`particle_distribution_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   HomogeneousAtmosphere
   HeterogeneousAtmosphereLegacy
   HeterogeneousAtmosphere
   MolecularAtmosphere
   ParticleLayer

**Particle distribution**

.. autosummary::
   :toctree: generated/autosummary

   ParticleDistribution

.. _sec-reference-scenes-biosphere:

Biosphere [eradiate.scenes.biosphere]
-------------------------------------
.. currentmodule:: eradiate.scenes.biosphere

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

.. _sec-reference-scenes-surface:

Surfaces [eradiate.scenes.surface]
----------------------------------
.. currentmodule:: eradiate.scenes.surface

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Surface

**Factories**

* :data:`surface_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   BlackSurface
   CheckerboardSurface
   LambertianSurface
   RPVSurface

.. _sec-reference-scenes-illumination:

Illumination [eradiate.scenes.illumination]
-------------------------------------------
.. currentmodule:: eradiate.scenes.illumination

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
   ConstantIllumination

.. _sec-reference-scenes-measure:

Measures [eradiate.scenes.measure]
----------------------------------
.. currentmodule:: eradiate.scenes.measure

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Measure
   DistantMeasure

**Factories**

* :data:`measure_factory`

**Sensor information data structure**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _core.SensorInfo

**Measure spectral configuration**

.. autosummary::
   :toctree: generated/autosummary/

   MeasureSpectralConfig

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _core.MonoMeasureSpectralConfig
      _core.CKDMeasureSpectralConfig

**Result storage and processing**

.. autosummary::
   :toctree: generated/autosummary/

   MeasureResults

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   DistantRadianceMeasure
   DistantReflectanceMeasure
   DistantFluxMeasure
   DistantAlbedoMeasure
   DistantArrayMeasure
   DistantArrayReflectanceMeasure
   PerspectiveCameraMeasure
   RadiancemeterMeasure
   MultiRadiancemeterMeasure

**Target and origin specification for DistantMeasure**

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _distant.TargetOrigin
      _distant.TargetOriginPoint
      _distant.TargetOriginRectangle
      _distant.TargetOriginSphere

.. _sec-reference-scenes-phase_functions:

Phase functions [eradiate.scenes.phase]
---------------------------------------
.. currentmodule:: eradiate.scenes.phase

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

.. _sec-reference-scenes-integrators:

Integrators [eradiate.scenes.integrators]
-----------------------------------------
.. currentmodule:: eradiate.scenes.integrators

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

.. _sec-reference-scenes-spectra:

Spectra [eradiate.scenes.spectra]
---------------------------------
.. currentmodule:: eradiate.scenes.spectra

**Interfaces**

.. autosummary::
   :toctree: generated/autosummary/

   Spectrum

**Factories**

* :data:`spectrum_factory`

**Scene elements**

.. autosummary::
   :toctree: generated/autosummary/

   UniformSpectrum
   InterpolatedSpectrum
   SolarIrradianceSpectrum
   AirScatteringCoefficientSpectrum
