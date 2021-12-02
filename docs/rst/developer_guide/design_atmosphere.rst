.. _sec-developer_guide-design_atmosphere:


Design note: atmosphere
=======================

Eradiate internally represents the atmosphere as a cuboid and the surface as a
rectangle. When positioning both shapes during kernel initialisation, they must
not overlap, *i.e.* be positioned at exactly the same altitude: ray-surface
intersection would otherwise become unpredictable and results would be wrong.

For this reason, Eradiate makes the shape delimiting the atmosphere slightly
larger than it actually should and offsets it so that the bottom of the
atmosphere's kernel shape does not exactly coincide with the surface.

In addition, the :class:`~eradiate.scenes.atmosphere.Atmosphere` interface
defines a series of attributes required to position the atmosphere's kernel
shape vertically:

* ``top`` is the altitude corresponding to the so-called "top of atmosphere"
  level;
* ``bottom`` is the altitude corresponding to the ground level;
* ``height`` is equal to ``top - bottom``;
* ``kernel_offset`` is the distance by which the atmosphere's kernel shape
  should be offset (towards lower altitudes) to prevent ray intersection issues
  at surface level;
* ``kernel_height`` is the height of the kernel shape and is equal to
  ``height + kernel_offset``.

.. only:: latex

   .. figure:: ../../fig/atmosphere_attributes.png

.. only:: not latex

   .. figure:: ../../fig/atmosphere_attributes.svg

