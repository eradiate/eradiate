.. _sec-user_guide-data-solar_irradiance:

Solar irradiance spectra
========================

A solar irradiance spectrum dataset provides the Sun's irradiance
spectrum at a Sun-Earth distance of 1 astronomical unit.
Solar spectral irradiance data may come from observations—*e.g.* using a
spectrometer onboard a satellite—or theoretical models such as the blackbody
model, or from a combination of observation data and theoretical models.

Data access
-----------

All solar irradiance spectrum datasets required by Eradiate are
managed the data store (see :ref:`sec-user_guide-data-intro` for details).

Identifier format
^^^^^^^^^^^^^^^^^

Identifiers for solar irradiance spectrum datasets (except ``blackbody_sun``)
are constructed based on the format ``{author}_{year}[-{extra}]`` where:

* ``author`` denotes the author of the data set,
* ``year`` stands for the year in which the data set was published,
* ``extra`` (optional) includes additional information, such as a dataset
  spectral resolution.

Structure
---------

Solar irradiance spectrum datasets include one data variable:

* the solar spectral irradiance (``ssi``)

and two :term:`dimension coordinates <dimension coordinate>`:

* the wavelength (``w``),
* the time (``t``).

Solar spectral irradiance data is tabulated against both wavelength and time.

Dataset metadata comply with the
`NetCDF Climate and Forecast (CF) Metadata Conventions
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html>`_
and helps with data traceability by providing the following attributes:

* ``history``: the history of transformations that the original data has undergone
* ``data_url``: the URL where the original data has been downloaded from
* ``data_url_datetime``: the date and time at which the original data has been downloaded

Available datasets
------------------

.. list-table:: Available datasets
   :widths: 25 25 25 25
   :header-rows: 1

   * - Identifier
     - Range
     - Sampling resolution
     - Reference
   * - ``blackbody_sun``
     - [280, 2400] nm
     - 0.1 nm
     - :cite:`Liou2002IntroductionAtmosphericRadiation`.
   * - ``coddington_2022-1_nm``
     - [202, 2730] nm
     - 1 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2022-0.1_nm``
     - [202, 2730] nm
     - 0.1 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2022-0.025_nm``
     - [202, 2730] nm
     - 0.025 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2022-0.005_nm``
     - [202, 2730] nm
     - 0.005 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2022-fse``
     - [0.115, 200] µm
     - 0.001 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2022-fse_binned``
     - [0.115, 200] µm
     - 1 nm
     - :cite:`Coddington2022TSIS1HSRSVersion2`
   * - ``coddington_2021-1_nm``
     - [202, 2730] nm
     - 1 nm
     - :cite:`Coddington2021TSIS1HSRS`
   * - ``coddington_2021-0.1_nm``
     - [202, 2730] nm
     - 0.1 nm
     - :cite:`Coddington2021TSIS1HSRS`
   * - ``coddington_2021-0.025_nm``
     - [202, 2730] nm
     - 0.025 nm
     - :cite:`Coddington2021TSIS1HSRS`
   * - ``coddington_2021-0.005_nm``
     - [202, 2730] nm
     - 0.005 nm
     - :cite:`Coddington2021TSIS1HSRS`
   * - ``meftah_2018``
     - [165.0, 3000.1] nm
     - :math:`\leq 1` nm
     - :cite:`Meftah2018SOLARISSReference`
   * - ``solid_2017``
     - [0.5, 1991.5] nm
     - :math:`\leq 16` nm
     - :cite:`Haberreiter2017ObservationalSolarIrradiance`
   * - ``solid_2017-mean``
     - [0.5, 1991.5] nm
     - :math:`\leq 16` nm
     - :cite:`Haberreiter2017ObservationalSolarIrradiance`
   * - ``whi_2008-sunspot_active``
     - [116.5, 2399.95] nm
     - 0.1 nm
     - :cite:`Woods2008SolarIrradianceReference`
   * - ``whi_2008-faculae_active``
     - [116.5, 2399.95] nm
     - 0.1 nm
     - :cite:`Woods2008SolarIrradianceReference`
   * - ``whi_2008-quiet_sun``
     - [116.5, 2399.95] nm
     - 0.1 nm
     - :cite:`Woods2008SolarIrradianceReference`
   * - ``thuillier_2003``
     - [200, 2397] nm
     - 1 nm
     - :cite:`Thuillier2003SolarSpectralIrradiance`
   * - ``thuillier_2003-extrapolated``
     - [200, 2500] nm
     - 1 nm
     - :cite:`Thuillier2003SolarSpectralIrradiance`

Below is a brief description of each of the available solar irradiance
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
280 nm to 2400 nm to cover Eradiate's wavelength range.

``coddington_2022-*``
^^^^^^^^^^^^^^^^^^^^^

This is the version 2 of the Total and Spectral Solar Irradiance Sensor-1
(TSIS-1) Hybrid Solar Reference Spectrum (HSRS).

* Wavelength range (in vacuum): 202 nm to 2730 nm
* Spectral resolution: 0.01 nm to ~0.001 nm (variants are also provided at lower, fixed, spectral resolution).
* Time range: representative of a 1-week average from Dec 1, 2019 to Dec 7, 2019
* Uncertainty:
    - :math:`\leq 400` nm: 1.3%
    - [400, 460] nm: 0.5%
    - [460, 2365] nm: 0.3%
    - :math:`\geq 2365` nm: 1.3%

``coddington_2022-fse*``
^^^^^^^^^^^^^^^^^^^^^^^^

This is the Full Spectrum Extension (FSE) of the version 2 of the Total and
Spectral Solar Irradiance Sensor-1 (TSIS-1) Hybrid Solar Reference Spectrum
(HSRS).

* Wavelength range (in vacuum): 0.115 µm to 200 µm
* Time range: representative of a 1-week average from Dec 1, 2019 to Dec 7, 2019
* Uncertainty:
    - [0.115, 0.202] µm: [2, 15] %
    - [0.202, 0.4] µm: 1.3%
    - [0.4≤0.46] µm: 0.5%
    - [0.46, 2.365] µm: 0.3%
    - [2.365, 2.73] µm: 1.3%
    - [2.73, 100] µm: [1, 8] %
    - [100, 200] µm: 8%

``coddington_2021-*``
^^^^^^^^^^^^^^^^^^^^^

The Total and Spectral Solar Irradiance Sensor-1 (TSIS-1) Hybrid Solar
Reference Spectrum (HSRS) combines data from the TSIS-1 Spectral Irradiance
Monitor (SIM), CubeSat Compact SIM (CSIM), Air Force Geophysical Laboratory
ultraviolet solar irradiance balloon observations, ground-based Quality
Assurance of Spectral Ultraviolet Measurements In Europe Fourier transform
spectrometer solar irradiance observations, Kitt Peak National Observatory
solar transmittance atlas and the semi-empirical Solar Pseudo-Transmittance
Spectrum atlas.

In March 2022, it was recommended as the new solar irradiance reference
spectrum standard by the Committee on Earth Observation Satellites (CEOS)
Working Group on Calibration and Validation (WGCV).

* Wavelength range (in vacuum): 202 nm to 2730 nm
* Spectral resolution: 0.01 nm to ~0.001 nm (variants are also provided at lower, fixed, spectral resolution).
* Time range: representative of a 1-week average from Dec 1, 2019 to Dec 7, 2019
* Uncertainty:
    - :math:`\leq 400` nm: 1.3%
    - [400, 460] nm: 0.5%
    - [460, 2365] nm: 0.3%
    - :math:`\geq 2365` nm: 1.3%

``meftah_2018``
^^^^^^^^^^^^^^^

A reference solar irradiance spectrum based on observations
from the SOLSPEC instrument of the SOLAR payload onboard the internationial
space station. The spectrum was built using observation data from 2008 for
the [165, 656] nm wavelength range and from 2010--2016 for the [656, 3000] nm
wavelength range. The spectrum is said to be representative of the 2008 solar
minimum which corresponds to the end of the solar cycle 23 and the beginning
of the solar cycle 24.

* Wavelength range: [165.0, 3000.1] nm.
* Resolution: better than 1 nm below 1000 nm, and 1 nm in the [1000, 3000] nm wavelength range.
* Absolute uncertainty: 1.26 % (1 standard deviation).
* Total solar irradiance: 1372.3 ± 16.9 W/m² (1 standard deviation).

``solid_2017``
^^^^^^^^^^^^^^

An observational solar irradiance spectrum composite based on
data from 20 different instruments. The dataset provides daily solar
irradiance spectra from 1978-11-7 to 2014-12-31.

* Wavelength range: [0.5, 1991.5] nm.
* Resolution: variable, between 1 and 16 nm.

See also
`the Cal/Val Portal of the Committee on Earth Observation Satellites
<http://calvalportal.ceos.org/solar-irradiance-spectrum>`_.

``solid_2017-mean``
^^^^^^^^^^^^^^^^^^^

A time-average of the ``solid_2017`` dataset over all days
from 1978-11-7 to 2014-12-31.

``whi_2008-*``
^^^^^^^^^^^^^^

A combination of simultaneous satellite observations from the
SEE and SORCE instruments (from 2008-03-25 to 2008-04-16) onboard the TIMED
satellite and a prototype EVE instrument onboard a sounding rocket launched
on 14 April 2008. Representative of solar cycle minimum conditions.

* Wavelength range: [116.5, 2399.95] nm (the wavelengths [0.5, 116.5] nm are cut off).
* Resolution: 0.1 nm.

The WHI campaign produced three spectra, corresponding to three time periods:

- ``whi_2008-sunspot_active``: from 2008-03-25 to 2008-03-29, "sunspot active" spectrum.
  Total solar irradiance: 1360.70 W/m².

- ``whi_2008-faculae_active``: from 2008-03-29 to 2008-04-4, "faculae active" spectrum.
  Total solar irradiance: 1360.94 W/m².

- ``whi_2008-quiet_sun``: from 2008-04-10 to 2008-04-16, "quiet sun" spectrum.
  Total solar irradiance: 1360.84 W/m².

``thuillier_2003``
^^^^^^^^^^^^^^^^^^

A reference solar irradiance spectrum based on observations
from the SOLSPEC instrument during the ATLAS-1 mission (from 1992-03-24 to
1992-04-02) and the SOSP instrument onboard the EURECA satellite
(from 1992-8-7 to 1993-7-1), and on the Kurucz and Bell (1995) synthetic
spectrum.

* Wavelength range: [200, 2397] nm.
* Resolution: 1 nm.

The mean absolute uncertainty is of 2 to 3 %. The spectrum is representative of
moderately high solar activity. When contributions from the wavelength region
:math:`[0, 200[ \, \cup \, ]2397, +\infty[` nm are added, the total solar
irradiance evaluates to 1367.7 W/m². In [200, 2397] nm, the integrated solar
irradiance spectrum evaluates to 1315.7 W/m².

``thuillier_2003-extrapolated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A version of the ``thuillier_2003`` spectrum extrapolated to 2500 nm so that it
covers the wavelength range from 200 to 2500 nm.
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
