.. _sec-developer_guide-scene_element_guide:

Writing a new scene element class
=================================

.. warning::

   The entire scene generation architecture is built using the
   `attrs <https://www.attrs.org>`_ library. It is strongly recommended to
   read the ``attrs`` documentation prior to writing a new element class. In
   particular, it is important to understand the ``attrs`` initialisation
   sequence, as well as how callables can be used to set defaults and to
   create converters and validators.

   In addition, Eradiate provides its own extensions to ``attrs`` located in the
   :mod:`eradiate.util.attrs` module. This guide will explain how these
   components are used.

   Finally, Eradiate's unit handling is based on
   `Pint <https://pint.readthedocs.io>`_, whose documentation is also a very
   helpful read.

Scene elements, deriving from the :class:`~eradiate.scenes.core.SceneElement`
class, are the core of Eradiate's scene generation system. They provide an
interface to quickly and safely generate kernel scene dictionary elements
(see :class:`~eradiate.scenes.core.KernelDict`).

The :class:`~eradiate.scenes.core.SceneElement` base class
----------------------------------------------------------

:class:`~eradiate.scenes.core.SceneElement` is the abstract base class for all
scene elements. We will see here how this class works, and then how to write a
new scene element subclass.

* :class:`~eradiate.scenes.core.SceneElement` is decorated by :func:`attr.s`.
  This allows ``attrs`` to work nicely with its derived classes.
  :class:`~eradiate.scenes.core.SceneElement` has an ``id`` instance attribute
  with a default value: consequently, all instance attributes defined for
  child classes must have default values.

* :class:`~eradiate.scenes.core.SceneElement` is decorated by
  :func:`~eradiate.util.attrs.unit_enabled`. This allows to mark attributes
  as having units upon definition. It also adds a ``from_dict()`` class method
  which allows for the instantiation of the decorated type with automatic
  handling of associated unit fields. Finally, it makes it possible to use the
  :func:`eradiate.util.attrs.attrib_quantity` helper to conveniently declare
  fields supporting :class:`pint.Quantity` containers.

  .. code-block:: python

     import attr
     import yaml
     from eradiate.scenes.core import SceneElement
     from eradiate.util.units import ureg
     from eradiate.util.attrs import attrib_quantity

     @attr.s
     class MyElement(SceneElement):
         # This is an ordinary field declaration
         a = attr.ib(default=1.)
         # This field will always have units compatible with ureg.m
         b = attrib_quantity(default=1., units_compatible=ureg.m)

         def kernel_dict(): ...  # Definition skipped

     # Our class can be instantiated from a dictionary
     obj = MyElement.from_dict({"a": 1., "b": ureg.Quantity(1., ureg.m)})
     # We can also specify the units of b in the dictionary
     obj = MyElement.from_dict({"a": 1., "b": 1., "b_units": "m"})
     # This is especially useful when creating objects from YAML files
     obj = MyElement.from_dict(yaml.safe_load("""
         a: 1.
         b: 100.
         b_units: cm
     """))

* :class:`~eradiate.scenes.core.SceneElement` works around unit quantities in
  a convenient fashion. Properly defined fields can be set using
  :class:`pint.Quantity` objects: if so, unit compatibility will be checked upon
  assignment and Eradiate will raise if units are found to be incompatible.

  .. code-block:: python

     import attr
     from eradiate.scenes.core import SceneElement
     from eradiate.util.units import ureg
     from eradiate.util.attrs import attrib_quantity

     @attr.s
     class MyElement(SceneElement):
         field = attrib_quantity(default=1., units_compatible=ureg.m)
         def kernel_dict(): ...  # Definition skipped

     # This is valid
     obj = MyElement(field=ureg.Quantity(1., ureg.m))
     # This will raise a UnitsError: second is not a distance unit
     obj = MyElement(field=ureg.Quantity(1., ureg.s))
     # This will raise a UnitsError: we check for units, not only for dimensionality
     obj = MyElement(field=ureg.Quantity(1., ureg.m / ureg.deg))

  If a unitless value is passed to a quantity field, it will be automatically
  added the compatible unit:

  .. code-block:: python

     import attr
     from eradiate.scenes.core import SceneElement
     from eradiate.util.units import ureg
     from eradiate.util.attrs import attrib_quantity

     @attr.s
     class MyElement(SceneElement):
         field = attrib_quantity(default=1., units_compatible=ureg.m)
         def kernel_dict(): ...  # Definition skipped

     # This is valid
     obj = MyElement(field=1.)
     assert obj.field == ureg.Quantity(1., ureg.m)
     # It also works when instantiating from dictionaries
     obj = MyElement.from_dict({"field": 1.})
     assert obj.field == ureg.Quantity(1., ureg.m)

* :class:`~eradiate.scenes.core.SceneElement` has a single abstract method
  :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` which must be
  implemented by its derived classes: it returns a dictionary which can be then
  used as an input to the kernel.

Constructing elements from the factory
--------------------------------------

The :class:`~eradiate.scenes.core.SceneElementFactory` class can be used to
construct registered :class:`~eradiate.scenes.core.SceneElement` derived classes.
Scene elements can be made accessible through Eradiate's factory system very
easily. The class definition simply has to be decorated using the
:meth:`SceneElementFactory.register() <eradiate.scenes.core.SceneElementFactory.register>`
decorator.

.. code-block:: python

   import attr
   from eradiate.scenes.core import SceneElement, SceneElementFactory
   from eradiate.util.units import ureg
   from eradiate.util.attrs import attrib_quantity

   @SceneElementFactory.register(name="my_element")
   @attr.s
   class MyElement(SceneElement):
       field = attrib_quantity(default=1., units_compatible=ureg.m)
       def kernel_dict(): ...  # Definition skipped

   obj = SceneElementFactory.create({"type": "my_element", "field": 1.})

At this point, it is also important to check if the module in which the element
to be registered is located is properly registered as a search location in the
:class:`~eradiate.scenes.core.SceneElementFactory` class. By default,
:class:`~eradiate.scenes.core.SceneElementFactory` holds of list of modules
where to search for factory-enabled classes; however, classes defined outside of
Eradiate's codebase won't be included in that list and it is the user's
responsibility to make sure that their custom element classes are imported at
some point so as to be registered to the factory.

Defining quantity fields
------------------------

.. warning::

   This section absolutely requires familiarity with the ``attrs`` `init
   sequence <https://www.attrs.org/en/stable/init.html#order-of-execution>`_ and
   associated concepts (default, validator, converter, factory).

As previously mentioned, the :func:`.attrib_quantity` helper function is
designed to automate the declaration of quantity fields. It wraps
:func:`attr.ib` and adds three parameters:


Parameter ``units_compatible`` (callable or :class:`pint.Unit` or str or None)
    This parameter sets the attribute's compatible units. If unset,
    :func:`.attrib_quantity` is just like :func:`attr.ib`. ``units_compatible``
    can either be a Pint unit (created from Eradiate's unit registry), **or a
    callable which will then be dynamically when relevant.**

   .. code-block:: python

      from eradiate.util.attrs import attrib_quantity
      from eradiate.util.units import ureg, config_default_units as cdu

      # Static default unit declaration
      field = attrib_quantity(units_compatible=ureg.m)
      # Dynamic default unit declaration: cdu.generator("length") returns a
      # callable which, when evaluated, returns configuration default length
      # units
      field = attrib_quantity(units_compatible=cdu.generator("length"))

Parameter ``units_add_converter`` (bool)
    This parameter is a boolean. If set to ``True`` (its default value),
    :func:`.attrib_quantity` adds a converter to the attribute's conversion
    pipeline. This converter transforms the current field value into a
    :class:`pint.Quantity` object using the value passed to
    ``units_compatible`` if it is unitless. If it is a callable, it is evaluated
    at the moment where the attribute is set. This leads to the following
    behaviour:

    .. code-block:: python

       import attr
       from eradiate.util.attrs import attrib_quantity, unit_enabled
       from eradiate.util.units import ureg, config_default_units as cdu

       @unit_enabled
       @attr.s
       class MyClass:
           field = attrib_quantity(
               default=ureg.Quantity(1, "m"),
               units_compatible=cdu.generator("length"),
               units_add_converter=True
           )

       with cdu.override({"length": "km"}):
           obj = MyClass(1.)
       assert obj.field == ureg.Quantity(1., "km")
       with cdu.override({"length": "m"}):
           obj.field = 1.
       assert obj.field == ureg.Quantity(1., "m")

    Sometimes, the automated addition of the converter will be inappropriate;
    in such cases, setting ``units_add_converter`` to ``False`` and manually
    defining the field's converter is the way to go.

Parameter ``units_add_validators`` (bool)
    If this boolean parameter is set to ``True`` (the default), then a validator
    rejecting values with incompatible units will be appended to the validation
    sequence.

    .. code-block:: python

       import attr
       from eradiate.util.attrs import attrib_quantity, unit_enabled
       from eradiate.util.units import ureg, config_default_units as cdu

       @unit_enabled
       @attr.s
       class MyClass:
           field = attrib_quantity(
               default=ureg.Quantity(1, "m"),
               units_compatible=cdu.generator("length"),
               units_add_converter=True,
               units_add_validator=True,
           )

       # This will fail: seconds are not compatible with metres
       obj = MyClass(ureg.Quantity(1, "s"))


    Sometimes, the automated addition of the validator will be inappropriate;
    in such cases, setting ``units_add_validator`` to ``False`` and manually
    defining the field's validator is the way to go.

In addition, :func:`.attrib_quantity` overrides the default of :func:`attr.ib`'s
``on_setattr`` parameter and, if unset, sets ``on_setattr`` to perform
conversion and validation. If :func:`.attrib_quantity`'s ``on_setattr`` is set,
the normal behaviour of :func:`attr.ib` is preserved.

Using factory converters
------------------------

The final piece of scene element writing is the use of factory converters. As
mentioned in the :ref:`sec-developer_guide-factory_guide`, Eradiate's factories
implement a :func:`~eradiate.util.factory.BaseFactory.convert` class method
which can turn a dictionary into a registered object—and if the method receives
something else than a dictionary, it simply does nothing.

This method can be used as a converter in the attribute initialisation sequence
to automatically convert a dictionary to a specified object. This allows for
the use of nested dictionaries to instantiate multiple objects.

.. code-block:: python
   :emphasize-lines: 17

   import attr

   from eradiate.scenes.core import SceneElement, SceneElementFactory
   from eradiate.util.attrs import attrib_quantity
   from eradiate.util.units import ureg

   @SceneElementFactory.register("element_a")
   @attr.s
   class ElementA(SceneElement):
       field = attr.ib(default=1.)
       def kernel_dict(): ...  # Definition skipped

   @SceneElementFactory.register("element_b")
   @attr.s
   class ElementB(SceneElement):
       element_a = attr.ib(
           default=ElementA(),
           converter=SceneElementFactory.convert
       )
       def kernel_dict(): ...  # Definition skipped

   # Pass object created with constructor
   obj = ElementB(element_a=ElementA(field=2.))
   # Use the factory to convert a dictionary to ElementA
   obj = ElementB(element_a={"type": "element_a", "field": 3.})
   # Instantiate ElementB using nested dicts
   obj = SceneElementFactory.create({
       "type": "element_b",
       "element_a": {"type": "element_a", "field": 4.}
   })
   # Same using YAML
   obj = SceneElementFactory.create(yaml.safe_load("""
       type: element_b
       element_a:
           type: element_a
           field: 4.
   """))

The :meth:`~.SceneElement.kernel_dict` method
---------------------------------------------

Any scene element **must** implement a :meth:`~.SceneElement.kernel_dict` method
which will return a kernel dictionary. These dictionaries are written following
the Mitsuba scene specification and the interested reader is referred to kernel
docs for further information.

.. note::

   When writing the :meth:`~.SceneElement.kernel_dict` method, there are a few
   precautions to keep in mind:

   * kernel imports must be local to the method;
   * if a kernel import is required to build the dictionary, a kernel variant
     must be selected when it is called (in practice, this means that Eradiate's
     operational mode must have been selected);
   * :meth:`~.SceneElement.kernel_dict`'s signature should allow for the
     processing of a ``ref`` argument, which, when set to ``True``, makes the
     method return object references when relevant (it is not always the case).

In practice: Steps to write a new scene element class
-----------------------------------------------------

Following the above description, a new scene element class requires the
following steps:

1. Derive a new class from :class:`~eradiate.scenes.core.SceneElement`. Decorate
   it with :func:`attr.s`.
2. Declare your custom attributes using :func:`attr.ib`. Don't forget to add
   default values to all of them. Use :func:`~eradiate.util.attrs.attrib_quantity`
   if the field represents a physical quantity with units. Callables can be used
   to evaluate units dynamically. If the field requires it, it is possible to
   run custom converters and validators.
3. Implement the :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method.
   Things to keep in mind:

   * kernel imports must be local to the
     :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method;
   * the function's signature should allow for the processing of a ``ref``
     keyword argument (but using it is not required).

The following steps are optional:

* implement a post-init hook steps using the ``__attrs_post_init__()`` method;
* enable factory-based instantiation using the
  :meth:`SceneElementFactory.register() <eradiate.scenes.core.SceneElementFactory.register>` decorator.
