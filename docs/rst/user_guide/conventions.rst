.. _sec-user_guide-conventions:

Conventions in Eradiate
=======================

Eradiate aims at building bridges between communities. In some cases, this can
however lead to difficult situations due to identical wording or names meaning
different things. For concepts, Eradiate has a glossary; for other things
bound to a convention (*e.g.* the practical definition of angles used in
spherical coordinates), this page should settle things.

Local coordinate definition
---------------------------

Internally, Eradiate and its kernel Mitsuba use Cartesian coordinates to locate
scene objects. Scene construction, when relevant, map the X, Y and Z axes to the
local East, North and up directions at the origin location. Note that this does
not mean that Eradiate always uses an East, North, up (ENU) coordinate system:
this is only relevant when planetary curvature can be neglected.

Local illumination and viewing angle definition
-----------------------------------------------

Internally, Eradiate defines the zenith angle :math:`\theta` as the angular
distance (in degree or radian) to the local zenith. From this, it follows that:

* the Sun is considered to be at the zenith when :math:`\theta_\mathrm{s} = 0°`;
* the sensor is considered to point towards the nadir when
  :math:`\theta_\mathrm{v} = 0°`.

The azimuth angle :math:`\varphi` is defined following the right-hand rule using the local
vertical, with its origin aligned with the local X axis (pointing towards East).
This, in practice, means that:

* :math:`\varphi = 0°` points towards East;
* :math:`\varphi = 90°` points towards North;
* :math:`\varphi = 180°` points towards West;
* :math:`\varphi = 270°` points towards South.

Principal plane orientation
---------------------------

Unless told otherwise, Eradiate indexes principal plane data using a signed
zenith angle in the [-90°, 90°] range, with the positive half-plane containing
the illumination direction. For this follows:

.. important::
   *On principal plane plots, the illumination is located to the right.*
