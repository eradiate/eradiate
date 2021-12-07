``eradiate.scenes``
===================

.. automodule:: eradiate.scenes

``eradiate.scenes.core``
------------------------

.. automodule:: eradiate.scenes.core

.. py:currentmodule:: eradiate.scenes.core

.. autosummary::
   :toctree: generated/autosummary/

   SceneElement
   KernelDict
   BoundingBox

``eradiate.scenes.atmosphere``
------------------------------

.. automodule:: eradiate.scenes.atmosphere

.. py:currentmodule:: eradiate.scenes.atmosphere

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
   HeterogeneousAtmosphere
   MolecularAtmosphere
   ParticleLayer

**Particle distribution**

.. autosummary::
   :toctree: generated/autosummary

   ParticleDistribution

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

   BlackSurface
   CentralPatchSurface
   CheckerboardSurface
   LambertianSurface
   RPVSurface

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
   ConstantIllumination

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

**Measure spectral configuration**

.. autosummary::
   :toctree: generated/autosummary/

   MeasureSpectralConfig

.. dropdown:: Private

   .. autosummary::
      :toctree: generated/autosummary/

      _core.MonoMeasureSpectralConfig
      _core.CKDMeasureSpectralConfig

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
   SolarIrradianceSpectrum
   AirScatteringCoefficientSpectrum
