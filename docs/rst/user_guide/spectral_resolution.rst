.. _sec-user_guide-spectral_resolution:

Simulation spectral resolution
##############################

.. note::
    
    This section assumes that you are familiar with :class:`Experiment` objects.


Depending on the active spectral mode, the spectral resolution has a different
meaning:

* In monochromatic mode, the spectral resolution is speficied by a
  :class:`WavelengthSet`, i.e., an array of wavelengths.
* In CKD mode, the spectral resolution is specified by a :class:`BinSet`, i.e.,
  a set of CKD bins.

An :class:`Experiment` has as many spectral resolutions as it has measures.

In both spectral modes, the spectral resolution cooresponding to each measure
is determined in three steps:

1. The spectral resolution is set to the :class:`Experiment` default
   spectral resolution, which is set by its :attr:`default_wset` and 
   :attr:`default_binset` attributes.
2. If an atmosphere is present and is molecular, the default spectral resolution
   is replaced by the spectral resolution of the atmosphere's absorption
   dataset.
3. A subset of either the :class:`BinSet` or the :class:`WavelengthSet` is
   selected to match the measure's :attr:`Measure.srf` attribute.

Hereafter we illustrate the spectral resolution selection process in common
use cases.

Monochromatic mode
******************

Without atmosphere
==================

Monochromatic simulation in a wavelength interval
-------------------------------------------------

This use case is typically encountered when the user wants to perform a
monochromatic simulation in the wavelength interval corresponding to the
spectral band of a given satellite instrument.
In this case, the user can simply set the :attr:`Measure.srf` attribute to the
identifier corresponding to the instrument's spectral band.

.. admonition:: Example

   The following example illustrates how to perform a monochromatic simulation
   in the wavelength interval corresponding to the 4th spectral band of the
   MODIS instrument onboard the AQUA platform, which extends from 538 nm to
   570 nm, using a default spectral resolution of 5 nm.
   In this example, the wavelength set consists of the following wavelengths:
   [535.0 540.0 545.0 550.0 555.0 560.0 565.0 570.0 575.0] nm. 
   To increase or decrease this spectral resolution, the user should set the
   :attr:`Experiment.default_wset` attribute to a different value.

   .. code-block:: python

      import eradiate
      from eradiate.experiments import AtmopshereExperiment

      exp = AtmosphereExperiment(
        default_wset=WavelengthSet.arange(
            280.0 * ureg.nm,
            2400.0 * ureg.nm,
            5.0 * ureg.nm,
        ),
        atmosphere=None,
        measures={
          "type": "multi_distant",
          "srf": "aqua-modis-4",
        }
      )

If you want to perform a monochromatic simulation in a wavelength interval
that does not correspond to a spectral band of a given satellite instrument,
you can use the :class:`InterpolatedSpectrum` class to define the spectral
response function that is going to select this wavelength interval.

.. admonition:: Example

   The following example illustrates how to perform a monochromatic simulation
   in the wavelength interval [540, 560] nm, at the default spectral resolution
   set by :attr:`Experiment.default_wset`.

   .. code-block:: python

      import eradiate
      from eradiate.experiments import AtmopshereExperiment
      from eradiate.scenes.spectra import InterpolatedSpectrum

      exp = AtmosphereExperiment(
        atmosphere=None,
        measures={
          "type": "multi_distant",
          "srf": InterpolatedSpectrum(
            wavelengths=np.array([540.0, 560.0]) * ureg.nm,
            values=np.array([1.0, 1.0]),
          )
        }
      )


Monochromatic simulation at isolated wavelengths
------------------------------------------------

The recommended way to achieve this is to use a :class:`MultiDeltaSpectrum` to 
define the spectral response function of the associated measure.
The wavelengths at which the simulation is performed are then specified by the
:attr:`MultiDeltaSpectrum.wavelengths` attribute.

.. admonition:: Example

   The following example illustrates how to perform a monochromatic simulation
   at 440 nm, 550 nm and 660 nm. 

   .. code-block:: python

      import numpy as np

      import eradiate
      from eradiate.experiments import AtmopshereExperiment
      from eradiate.scenes.spectra import MultiDeltaSpectrum
      from eradiate.units import unit_registry as ureg

      exp = AtmosphereExperiment(
        atmosphere=None,
        measures={
          "type": "multi_distant",
          "srf": MultiDeltaSpectrum(
            wavelengths=np.array([440.0, 550.0, 660.0]) * ureg.nm, 
          ),
        }
      )


With atmosphere
===============

When a molecular atmosphere is present, the :class:`Experiment` default
wavelength set is replaced by that of the atmosphere's absorption dataset. 

Monochromatic simulation in a wavelength interval
-------------------------------------------------

.. admonition:: Example

   The following example illustrates how to perform a monochromatic simulation
   at 440 nm, 550 nm and 660 nm. 

   .. code-block:: python

      import numpy as np

      import eradiate
      from eradiate.experiments import AtmopshereExperiment
      from eradiate.scenes.spectra import MultiDeltaSpectrum
      from eradiate.units import unit_registry as ureg

      exp = AtmosphereExperiment(
        atmosphere=None,
        measures={
          "type": "multi_distant",
          "srf": MultiDeltaSpectrum(
            wavelengths=np.array([440.0, 550.0, 660.0]) * ureg.nm, 
          ),
        }
      )

CKD mode
********

Without atmosphere
==================

With atmosphere
===============

