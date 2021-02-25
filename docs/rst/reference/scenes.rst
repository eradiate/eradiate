.. _sec-reference-scenes:

Scene generation [eradiate.scenes]
==================================

Core [eradiate.scenes.core]
---------------------------
.. currentmodule:: eradiate.scenes.core

.. autosummary::
   :toctree: generated/

   SceneElement
   KernelDict

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

Measures [eradiate.scenes.measure]
----------------------------------
.. currentmodule:: eradiate.scenes.measure

**Interfaces and factories**

.. autosummary::
   :toctree: generated/

   MeasureFactory
   Measure
   TargetOrigin

**Scene elements**

.. autosummary::
   :toctree: generated/

   DistantMeasure
   PerspectiveCameraMeasure
   RadianceMeterHsphereMeasure
   RadianceMeterPlaneMeasure

**Target and origin specification**

.. autosummary::
   :toctree: generated/

   TargetOriginPoint
   TargetOriginRectangle
   TargetOriginSphere

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
