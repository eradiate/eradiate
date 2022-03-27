.. _sec-user_guide-conventions:

Conventions in Eradiate
=======================

Eradiate aims at building bridges between scientific communities. In some cases,
this can however lead to difficult situations due to identical expressions or
names having a different meaning depending on context. For concepts, Eradiate
has a :ref:`glossary <sec-user_guide-basic_concepts>`; for other things bound to
a convention (*e.g.* the practical definition of angles used in spherical
coordinates), this page should settle things.

Local coordinate definition
---------------------------

.. only:: not latex

   .. margin::

      .. figure:: ../../fig/cartesian-coordinate-system.svg
         :width: 100%

Internally, Eradiate and its radiometric kernel Mitsuba use Cartesian
coordinates to locate scene objects. Scene construction, when relevant, map the
X, Y and Z axes to the local East, North and up directions at the origin
location.

.. only:: latex

   .. image:: ../../fig/cartesian-coordinate-system.png

Note that this does not mean that Eradiate always uses an East, North,
up (ENU) coordinate system: this is only relevant when planetary curvature can
be neglected.

Spherical coordinates
---------------------

.. only:: not latex

   .. margin::

      .. figure:: ../../fig/spherical-coordinate-system.svg
         :width: 100%

Internally, Eradiate also uses the spherical coordinate system with the
ISO 80000-2:2019 convention :cite:`ISO201980000QuantitiesUnits` (commonly used in
physics) where:

* :math:`r`, denotes the radial distance whose magnitude is a positive number,
* :math:`\theta \in [0, \pi]` rad, equivalently :math:`[0, 180]°`, denotes the
  zenith angle, a.k.a. the colatitude, polar angle, normal angle or
  inclination angle, which measures the angular distance to the local zenith,
* :math:`\varphi \in [0, 2\pi[` rad, equivalently :math:`[0, 360[°`, denotes
  the azimuth angle.

.. only:: latex

   .. image:: ../../fig/spherical-coordinate-system.png

Local illumination and viewing angle definition
-----------------------------------------------

.. only:: not latex

   .. margin::

      .. figure:: ../../fig/azimuth-angle-convention.svg
         :width: 100%

The directions to the Sun and to the sensor are specified in the
:math:`(\theta, \varphi)` space.
In our convention, it follows that:

* the Sun is considered to be at the zenith when :math:`\theta_\mathrm{s} = 0°`;
* the sensor is considered to point towards the nadir when
  :math:`\theta_\mathrm{v} = 0°`.

The azimuth angle :math:`\varphi` is defined following the right-hand rule using
the local vertical, with its origin aligned with the local X axis (pointing
towards East), as illustrated below.

.. only:: latex

   .. image:: ../../fig/azimuth-angle-convention.png


Principal plane orientation
---------------------------

Unless told otherwise, Eradiate indexes principal plane data using a signed
zenith angle in the [-90°, 90°] range, with the positive half-plane containing
the illumination direction. From this follows:

.. important::

   *On principal plane plots, the illumination is located to the right.*
