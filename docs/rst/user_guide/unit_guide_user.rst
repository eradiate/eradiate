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

This guide presents how unit handling is documented in the API reference and how
to handle units in Eradiate it as a user. For a technical view on unit handling,
see :ref:`sec-developer_guide-unit_guide_developer`.

.. note:: 

   It is strongly advised to—at least—get familiar with
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
require a user to specify a unit—it's so obvious!

.. admonition:: Example

   A class creates a tree object in a scene. The tree is parametrised by its
   height, and it's obvious that it should be specified in metres. Therefore, a
   user would most certainly like that a call like

   .. code-block:: python

      Tree(height=3.)

   creates a tree with a height of 3 metres.

However, it can happen that the "obvious" unit may vary depending on context.
For instance, an atmosphere can be kilometre-sized; however, it is common
practice to use metres to express altitude values. What is the "obvious" unit,
then? Some user could say it is the kilometre, another might prefer using the
metre. For this reason, Eradiate keeps track of default units used to configure
its objects. This is called the *configuration default unit set*, used to attach
units to dimensional quantities when the user does not specify them. By default,
default configuration units are the SI units; but a user can override them if it
seems more convenient in their context.

.. admonition:: Example

   .. code-block:: python

      import eradiate
      from eradiate.scenes.atmosphere import RayleighHomogeneousAtmosphere
      from eradiate.util.units import config_default_units as cdu

      eradiate.set_mode("mono")

      # This sets the configuration default length unit to kilometre
      with cdu.override({"length": "km"}):
          my_atmosphere = RayleighHomogeneousAtmosphere(height=100.)

   This will create a :class:`.RayleighHomogeneousAtmosphere` object with a
   height of 100 km. It is equivalent to the default

   .. code-block:: python

      my_atmosphere = RayleighHomogeneousAtmosphere(height=100e3)

In case of doubt, Eradiate allows to specify the unit used for internal
representation of quantities in a object. Each unit-enabled field has a
corresponding unit field which bears the same name and a `_unit` suffix:
Eradiate stores the magnitude and units of a quantity in two distinct fields.
The technical motivation for this design is given in
:ref:`sec-developer_guide-scene_element_guide`.

.. admonition:: Example

   The following code snippet will use kilometres to represent the height of the
   atmosphere without changing configuration default units:

   .. code-block:: python

      import eradiate
      from eradiate.scenes.atmosphere import RayleighHomogeneousAtmosphere
      
      eradiate.set_mode("mono")
      my_atmosphere = RayleighHomogeneousAtmosphere(height=100., height_units="km")

   The internal representation will be 100 km.

Finally, a user may want to not modify configuration default units but still
specify units for added safety. Many of Eradiate's objects support Pint
quantities and will check that values assigned to their attributes have
appropriate units.
**Note that all quantities should be created using Eradiate's unit registry**
:data:`eradiate.util.units.ureg`.

.. admonition:: Example

   The following code snippet will use metres to represent the height of the
   atmosphere but the specification will be in kilometres:

   .. code-block:: python

      import eradiate
      from eradiate.util.units import ureg
      from eradiate.scenes.atmosphere import RayleighHomogeneousAtmosphere
      
      eradiate.set_mode("mono")
      my_atmosphere = RayleighHomogeneousAtmosphere(height=ureg.Quantity(100., "km"))

   If one tries to set ``height`` with a value which has wrong units, a
   :class:`.UnitsError` will be raised:

   .. code-block:: python

      my_atmosphere.toa_altitude = ureg.Quantity(100., "s")  # This will raise a UnitsError

.. _sec-user_guide-unit_guide_user-field_unit_documentation:

Field unit documentation
------------------------

Eradiate documents fields with a unit by mentioning them as *unit-enabled*. All
unit-enabled fields have an associated unit field with a default value. Default
units are always created using Eradiate's unit registry. They can be fixed: in
that case, the unit will be given directly in the documentation. Default units
can also be dynamically selected at runtime by the user through the default unit
sets. In that case, the default unit is documented with a string with the
following structure: ``<unit_set>[<quantity>]`` where

* ``<unit_set>`` is either ``cdu`` for configuration default units or ``kdu``
  for kernel default units;
* ``<quantity>`` is the physical quantity ID used to query the default unit set
  (see :class:`.DefaultUnits` for a list of available quantity IDs).

Units fetching their defaults at runtime from default unit sets can be
overridden using :meth:`.DefaultUnits.override`.
