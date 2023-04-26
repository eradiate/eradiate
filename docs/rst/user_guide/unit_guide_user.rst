.. _sec-user_guide-unit_guide_user:

Unit handling for users
=======================

Eradiate tries hard to handle physical quantities correctly and uses the
`Pint unit handling library <https://pint.readthedocs.io>`_ to do so.
A general reason for that is that poor unit handling can lead to
`actual fiascos <https://pint.readthedocs.io/en/stable/#one-last-thing>`_.
Another reason, more specific to Eradiate, is that this enables for dynamic
quantity conversion, making it possible:

* for users: to specify the units in which they want to provide input;
* for developers: to change units with which they work (*e.g.* to scale scene
  dimensions dynamically).

.. important:: Under the hood, most unit handling facilities are implemented as
   part of the standalone `Pinttrs <https://pinttrs.readthedocs.io/>`_ library.

This guide presents how unit handling is documented in the API reference and how
to handle units in Eradiate as a user. For a technical view on unit handling,
see the Pinttrs documentation and source code.

.. note::  It is strongly advised to—at least—get familiar with
   `Pint <https://pint.readthedocs.io/>`_ to fully take advantage of Eradiate's
   unit support.


Units in Eradiate
-----------------

Eradiate usually works with the
`International System of units <https://en.wikipedia.org/wiki/International_System_of_Units>`_.
However, it also understands that users might use a different unit system.
Finally, Eradiate knows that it can be better to specify kernel scenes using yet
another unit system.

When a quantity is implicitly known as dimensional, it can be superfluous to
require a user to specify units—it's so obvious!

.. admonition:: Example
   :class: tip

   A class creates a tree object in a scene. The tree is parametrised by its
   height, and it's obvious that it should be specified in metres. Therefore, a
   user would most certainly like that a call like

   .. code-block:: python

      Tree(height=3.0)

   creates a tree with a height of 3 metres.

However, it can happen that the "obvious" unit may vary depending on context.
For instance, an atmosphere can be kilometre-sized; however, it is common
practice to use metres to express altitude values. What is the "obvious" unit,
then? Some user could say it is the kilometre, another might prefer using the
metre. For this reason, Eradiate keeps track of default units used to configure
its objects using a Pinttrs :class:`~pinttrs.UnitContext` instance. This
*configuration unit context* (:data:`eradiate.unit_context_config`, abbreviated
as ``ucc``) is used to attach units to dimensional quantities when the user does
not specify them. Default configuration units are the SI units; but
a user can override them if it seems more convenient in their context.

.. admonition:: Example
   :class: tip

   .. doctest::

      >>> import eradiate
      >>> from eradiate import unit_context_config as ucc
      >>> from eradiate.scenes.atmosphere import HomogeneousAtmosphere
      >>> eradiate.set_mode("mono")
      >>> with ucc.override(length="km"):  # Temporarily set default length unit to kilometre
      ...     my_atmosphere = HomogeneousAtmosphere(top=100.0)
      >>> my_atmosphere.top
      <Quantity(100.0, 'kilometer')>

   This will create a :class:`.HomogeneousAtmosphere` object with a
   height of 100 km. It is equivalent to the default

   .. doctest::

      >>> my_atmosphere = HomogeneousAtmosphere(top=100e3)
      >>> my_atmosphere.top
      <Quantity(100000.0, 'meter')>

Fields specified as "unit-enabled" are stored as Pint :class:`~pint.Quantity`
objects and can be passed as such. Eradiate ships its own unit registry
(:data:`eradiate.unit_registry`, abbreviated as ``ureg``), which must be used to
define units. Another way of initialising our 100 km-high atmosphere would then
be

.. doctest::

   >>> from eradiate import unit_registry as ureg
   >>> my_atmosphere = HomogeneousAtmosphere(top=100.0 * ureg.km)
   >>> my_atmosphere.top
   <Quantity(100.0, 'kilometer')>

If one tries to set ``top`` with a value which has wrong units, a
:class:`~pinttr.exceptions.UnitsError` will be raised:

.. doctest::

   >>> HomogeneousAtmosphere(top=100.0 * ureg.s)
   Traceback (most recent call last):
       ...
   pint.errors.DimensionalityError: Cannot convert from 'kilometer' ([length]) to 'second' ([time])

.. _sec-user_guide-unit_guide_user-field_unit_documentation:

Field unit documentation
------------------------

Eradiate documents fields with units by mentioning them as *unit-enabled*.
For those fields, automatic conversion of unitless values is implemented.
Default units can be fixed (*i.e.* invariant): in that case, units will be
specified directly in the documentation. Default units can also be dynamically
selected at runtime by the user through Eradiate's configuration unit context:
in that case, default units are documented with a string with the
following structure: ``<unit_context>[<quantity>]`` where

* ``<unit_context>`` is either ``ucc`` for configuration unit context or ``uck``
  for kernel unit context;
* ``<quantity>`` is the physical quantity ID used to query the default unit set
  (see :class:`~eradiate.units.PhysicalQuantity` for a list of available
  quantity IDs).

Units fetching their defaults at runtime from unit contexts can be
overridden using the :meth:`pinttrs.UnitContext.override` method.
