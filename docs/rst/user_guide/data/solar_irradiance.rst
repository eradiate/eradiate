.. _sec-user_guide-data-solar_irradiance:

Solar irradiance
================

A Solar irradiance spectrum data set provides the Sun's spectral irradiance
spectrum at a Sun-Earth distance of 1 astronomical unit.
Solar irradiance spectrum data may come from observations—*e.g.* using a
spectrometer onboard a satellite—or models such as the blackbody model.

Data access
-----------

All Solar irradiance spectrum data sets required by Eradiate are
managed the data store (see :ref:`sec-user_guide-data-intro` for details).

Identifier format
^^^^^^^^^^^^^^^^^

Identifiers for Solar irradiance spectrum data sets (except ``blackbody_sun``)
are constructed based on the format ``{author}_{year}[_{extra}]`` where:

* ``author`` denotes the author of the data set,
* ``year`` stands for the year in which the data set was published,
* ``extra`` (optional) includes additional information, such as a data set
  post-processing operation.

Structure
---------

Solar irradiance spectrum data sets include one data variable:

* the solar spectral irradiance (``ssi``)

and two :term:`dimension coordinates <dimension coordinate>`:

* the wavelength (``w``),
* the time (``t``).

Solar spectral irradiance data is tabulated against both wavelength and time.

Some of the following additional data set attributes are provided for data
sets to which they apply:

* ``obs_start``: observation start time,
* ``obs_end``: observation end time,
* ``url``: original data URL,
* ``comment``: a comment indicating how the original data was processed.

Description of available data sets
----------------------------------

Here is a quick description of each of the available solar irradiance
spectrum datasets.

``blackbody_sun``
^^^^^^^^^^^^^^^^^

A theoretical irradiance spectrum, based on Planck's law
for the blackbody spectral radiance:

.. math::

   L_{\lambda}(T) = \frac{2hc^2}{\lambda^5 (e^{hc/k\lambda T} - 1)}

where :math:`h` and :math:`k` are the Planck and Boltzmann constants
respectively, :math:`c` is the speed of light in a vacuum, :math:`\lambda` is
the wavelength, with a blackbody temperature :math:`T` of 5800 K—which is
roughly the temperature of the Sun's photosphere. The envelope of the Sun's
irradiance spectrum approximates that of a blackbody radiator. While converting
from spectral radiance to spectral irradiance, using the equation:

.. math::

   \phi_{\lambda}(T) = \frac{\pi R^2}{D^2} L_{\lambda} (T)

the radius of the blackbody (:math:`R`) is set to the solar radius constant
(:math:`695.7 \cdot 10^6` km) and the distance of the blackbody to the Earth
(:math:`D`) is set to 1 astronomical unit (:math:`149.5978707 \cdot 10^6` km)
which is the average Sun-Earth distance. The wavelength range extends from
280 nm to 2400 nm to cover Eradiate's wavelength range. Reference:
:cite:`Liou2002IntroductionAtmosphericRadiation`.

``meftah_2017``
^^^^^^^^^^^^^^^

A reference solar irradiance spectrum based on observations
from the SOLSPEC instrument of the SOLAR payload onboard the internationial
space station. The spectrum was built using observation data from 2008 for
the [165, 656] nm wavelength range and from 2010--2016 for the [656, 3000] nm
wavelength range. The spectrum is said to be representative of the 2008 solar
minimum which corresponds to the end of the solar cycle 23 and the beginning
of the solar cycle 24. Wavelength range: [165.0, 3000.1] nm. Resolution:
better than 1 nm below 1000 nm, and 1 nm in the [1000, 3000] nm wavelength
range. Absolute uncertainty: 1.26 % (1 standard deviation). Total solar
irradiance: 1372.3 ± 16.9 W/m² (1 standard deviation). Reference:
:cite:`Meftah2017SOLARISSReference`.

``solid_2017``
^^^^^^^^^^^^^^

An observational solar irradiance spectrum composite based on
data from 20 different instruments. The dataset provides daily solar
irradiance spectra from 1978-11-7 to 2014-12-31. Wavelength range: [0.5,
1991.5] nm. Resolution: variable, between 1 and 16 nm. Reference:
:cite:`Haberreiter2017ObservationalSolarIrradiance`. See also
`the Cal/Val Portal of the Committee on Earth Observation Satellites
<http://calvalportal.ceos.org/solar-irradiance-spectrum>`_.

``solid_2017_mean``
^^^^^^^^^^^^^^^^^^^

A time-average of the ``solid_2017`` dataset over all days
from 1978-11-7 to 2014-12-31.

``thuillier_2003``
^^^^^^^^^^^^^^^^^^

A reference solar irradiance spectrum based on observations
from the SOLSPEC instrument during the ATLAS-1 mission (from 1992-03-24 to
1992-04-02) and the SOSP instrument onboard the EURECA satellite
(from 1992-8-7 to 1993-7-1), and on the Kurucz and Bell (1995) synthetic
spectrum. Wavelength range: [200, 2397] nm. Resolution: 1 nm. The mean
absolute uncertainty is of 2 to 3 %. The spectrum is representative of
moderately high solar activity. Total solar irradiance: 1367.7 W/m².
Reference: :cite:`Thuillier2003SolarSpectralIrradiance`.

``thuillier_2003_extrapolated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A version of the ``thuillier_2003`` spectrum extrapolated to 2500 nm so that it
covers Eradiate's wavelength range.
The figure below illustrates the original and extrapolated versions and 
highlights the extrapolation region.

.. only:: latex

   .. image:: ../../../fig/data/data/srf/thuillier_2003_extrapolated.png

.. only:: not latex

   .. image:: ../../../fig/data/data/srf/thuillier_2003_extrapolated.svg

.. note::
  
   For the reference, we provide below the values of the integrated original 
   and extrapolated solar irradiance spectra, evaluated by integrating the
   irradiance spectrum along wavelength using the trapezoidal rule.

   .. math::

     \int_{200 \, \mathrm{nm}}^{2397 \, \mathrm{nm}}
     I_{\mathrm{original}} (\lambda) \, \mathrm{d} \lambda
     = 1315.68 \, \mathrm{W / m^2} \\

     \int_{200 \, \mathrm{nm}}^{2500 \, \mathrm{nm}}
     I_{\mathrm{extrapolated}} (\lambda) \, \mathrm{d} \lambda
     = 1321.72 \, \mathrm{W / m^2}
    
   Since the wavelength range is larger for the extrapolated irradiance
   spectrum, the corresponding integrated solar irradiance is also larger
   (by 0.46 %).

``whi_2008_*``
^^^^^^^^^^^^^^

A combination of simultaneous satellite observations from the
SEE and SORCE instruments (from 2008-03-25 to 2008-04-16) onboard the TIMED
satellite and a prototype EVE instrument onboard a sounding rocket launched
on 14 April 2008. Wavelength range: [116.5, 2399.95] nm (the wavelengths
[0.5, 116.5] nm are cut off). Resolution: 0.1 nm. Representative of solar cycle
minimum conditions. The WHI campaign produced three spectra, corresponding to
three time periods (numbered 1, 2, 3 here):

- ``whi_2008_1``: from 2008-03-25 to 2008-03-29, "sunspot active" spectrum.
  Total solar irradiance: 1360.70 W/m².

- ``whi_2008_2``: from 2008-03-29 to 2008-04-4, "faculae active" spectrum.
  Total solar irradiance: 1360.94 W/m².

- ``whi_2008_3``: from 2008-04-10 to 2008-04-16, "quiet sun" spectrum.
  Total solar irradiance: 1360.84 W/m².

``whi_2008`` is an alias to the quiet sun spectrum ``whi_2008_3``.
Reference: :cite:`Woods2008SolarIrradianceReference`.
