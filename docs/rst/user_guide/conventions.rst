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
coordinates to locate scene objects. When relevant, during scene construction,
the X, Y and Z axes are mapped to the local East, North and up directions at the
origin location.

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
ISO 80000-2:2019 convention :cite:`ISO201980000QuantitiesUnits` (commonly used
in physics) where:

* :math:`r`, denotes the radial distance whose magnitude is a positive number;
* `\theta \in [0, \pi]` rad, *i.e.* :math:`[0, 180]°`, denotes the
  *zenith angle*, a.k.a. the colatitude, polar angle, normal angle or
  inclination angle, which measures the angular distance to the local zenith;
* :math:`\varphi \in [0, 2\pi[` rad, *i.e.* :math:`[0, 360[°`, denotes the
  *azimuth angle*, defined with the X axis as its references and incremented
  counter-clockwise.

.. only:: latex

   .. image:: ../../fig/spherical-coordinate-system.png

Local illumination and viewing angle definition
-----------------------------------------------

The directions to the Sun and sensor are specified in the
:math:`(\theta, \varphi)` space. In our convention, it follows that:

* the Sun is considered to be at the zenith when :math:`\theta_\mathrm{s} = 0°`;
* the sensor is considered to point towards the nadir when
  :math:`\theta_\mathrm{v} = 0°`.

The azimuth angle :math:`\varphi` is defined following the right-hand rule using
the local vertical, with its origin aligned with the local X axis (pointing
towards East). We name this convention "East right". Other conventions are
frequently used in Earth observation applications, and Eradiate therefore
provides utility components to handle the conversion automatically.

.. _sec-user_guide-conventions-azimuth:

Azimuth definition conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Summary

   Eradiate works internally with the *East right* convention, which aligns the
   0° azimuth with the X axis and orients azimuth values counter-clockwise. If
   you work with geophysical data, chances are that you will prefer the
   *North left* convention, which aligns the 0° with the Y axis and increments
   azimuth values clockwise.

For convenience, Eradiate defines a set of azimuth definition conventions,
exposed to the user through a dedicated interface. All azimuth angle conventions
are defined, using the aforementioned East right convention as a reference, by:

* an `offset` value, which defines the direction with which the azimuth origin
  is aligned (incremented counter-clockwise);
* an `orientation`, which defines azimuth angles are counted positively
  clockwise or counter-clockwise (after applying the offset).

The offset is expressed in angle units (internally, radian) and the most common
values are aliased by the orientation corresponding to cardinal points on maps
in the Western tradition.

The two possible orientations are named after the right-hand rule: "right" is
counter-clockwise and corresponds to the + sign; "left" is clockwise and
corresponds to the - sign.

.. only:: not latex

   .. list-table:: Eradiate's built-in azimuth conventions.
      :align: center
      :header-rows: 1
      :stub-columns: 1
      :widths: 1 4 4

      * -
        - Right
        - Left

      * - East
        - .. image:: ../../fig/azimuth-east_right.svg
        - .. image:: ../../fig/azimuth-east_left.svg

      * - North
        - .. image:: ../../fig/azimuth-north_right.svg
        - .. image:: ../../fig/azimuth-north_left.svg

      * - West
        - .. image:: ../../fig/azimuth-west_right.svg
        - .. image:: ../../fig/azimuth-west_left.svg

      * - South
        - .. image:: ../../fig/azimuth-south_right.svg
        - .. image:: ../../fig/azimuth-south_left.svg

.. only:: latex

   .. list-table:: Eradiate's built-in azimuth conventions.
      :align: center
      :header-rows: 1
      :stub-columns: 1
      :widths: auto

      * -
        - Right
        - Left

      * - East
        - .. image:: ../../fig/azimuth-east_right.png
        - .. image:: ../../fig/azimuth-east_left.png

      * - North
        - .. image:: ../../fig/azimuth-north_right.png
        - .. image:: ../../fig/azimuth-north_left.png

      * - West
        - .. image:: ../../fig/azimuth-west_right.png
        - .. image:: ../../fig/azimuth-west_left.png

      * - South
        - .. image:: ../../fig/azimuth-south_right.png
        - .. image:: ../../fig/azimuth-south_left.png

Objects and functions taking angular parameters provide, when relevant, an
option to specify which convention is used. Manual conversion can be performed
using the :func:`eradiate.frame.transform_azimuth` function.

Principal plane orientation
---------------------------

Unless told otherwise, Eradiate indexes principal plane data using a signed
zenith angle in the [-90°, 90°] range, with the positive half-plane containing
the illumination direction. From this follows:

.. important::

   *On principal plane plots, the illumination is located to the right.*
