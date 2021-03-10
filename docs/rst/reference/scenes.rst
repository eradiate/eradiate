.. _sec-reference-scenes:

Scene generation [eradiate.scenes]
==================================

.. _sec-reference-scenes-core:

Core [eradiate.scenes.core]
---------------------------
.. currentmodule:: eradiate.scenes.core

.. autosummary::
   :toctree: generated/

   SceneElement
   KernelDict

.. _sec-reference-scenes-atmosphere:

Atmosphere [eradiate.scenes.atmosphere]
---------------------------------------
.. currentmodule:: eradiate.scenes.atmosphere

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   AtmosphereFactory
   Atmosphere

**Scene elements**

.. autosummary::
   :toctree: generated/

   HomogeneousAtmosphere
   HeterogeneousAtmosphere

.. _sec-reference-scenes-biosphere:

Biosphere [eradiate.scenes.biosphere]
-------------------------------------
.. currentmodule:: eradiate.scenes.biosphere

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   BiosphereFactory
   Canopy

**Scene elements**

.. autosummary::
   :toctree: generated/

   HomogeneousDiscreteCanopy

.. _sec-reference-scenes-surface:

Surfaces [eradiate.scenes.surface]
----------------------------------
.. currentmodule:: eradiate.scenes.surface

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   SurfaceFactory
   Surface

**Scene elements**

.. autosummary::
   :toctree: generated/

   BlackSurface
   LambertianSurface
   RPVSurface

.. _sec-reference-scenes-illumination:

Illumination [eradiate.scenes.illumination]
-------------------------------------------
.. currentmodule:: eradiate.scenes.illumination

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   IlluminationFactory
   Illumination

**Scene elements**

.. autosummary::
   :toctree: generated/

   DirectionalIllumination
   ConstantIllumination

.. _sec-reference-scenes-measure:

Measures [eradiate.scenes.measure]
----------------------------------
.. currentmodule:: eradiate.scenes.measure

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   MeasureFactory
   Measure

.. dropdown:: **Private: Sensor information data structure**

   .. autosummary::
      :toctree: generated/

      _core.SensorInfo

**Scene elements**

.. autosummary::
   :toctree: generated/

   DistantMeasure
   PerspectiveCameraMeasure
   RadiancemeterMeasure

.. dropdown:: **Private: Target and origin specification for DistantMeasure**

   .. autosummary::
      :toctree: generated/

      _distant.TargetOrigin
      _distant.TargetOriginPoint
      _distant.TargetOriginRectangle
      _distant.TargetOriginSphere

.. _sec-reference-scenes-integrators:

Integrators [eradiate.scenes.integrators]
-----------------------------------------
.. currentmodule:: eradiate.scenes.integrators

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   IntegratorFactory
   Integrator

**Scene elements**

.. autosummary::
   :toctree: generated/

   PathIntegrator
   VolPathIntegrator
   VolPathMISIntegrator

.. _sec-reference-scenes-spectra:

Spectra [eradiate.scenes.spectra]
---------------------------------
.. currentmodule:: eradiate.scenes.spectra

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   SpectrumFactory
   Spectrum

**Scene elements**

.. autosummary::
   :toctree: generated/

   UniformSpectrum
   SolarIrradianceSpectrum
