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
   components are to be used.

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
  This allows ``attrs`` to works nicely with its derived classes.

* :class:`~eradiate.scenes.core.SceneElement` is decorated by
  :func:`~eradiate.util.attrs.unit_enabled`. This activates a unit field
  association mechanism: each attribute defined with Eradiate's special
  :func:`~eradiate.util.attrs.attrib` wrapper (around :func:`attr.ib`) accepts a
  ``has_unit`` argument which makes it in turn mandatory to define an associated
  unit attribute. A :func:`~eradiate.util.attrs.unit_enabled` class checks upon
  declaration if all its attributes declared as having units indeed have an
  associated unit field with the same named suffixed with ``_unit``. For
  convenience, Eradiate provides the :func:`~eradiate.util.attrs.attrib_unit`
  wrapper which allows for convenient unit field definition.

  .. code-block:: python

     from eradiate.scenes.core import SceneElement
     from eradiate.util.units import ureg
     from eradiate.util.attrs import attrib, attrib_unit

     @attr.s
     class MyElement(SceneElement):
         # This field *must* have a corresponding unit field
         field = attrib(default=1., has_unit=True)
         # This is the corresponding unit field
         field_unit = attrib_unit(compatible_units=ureg.m, default=ureg.m)

         ...  # Rest of definition skipped

* :class:`~eradiate.scenes.core.SceneElement` works around unit quantities in
  a convenient fashion. While all attributes are meant to be unitless and unit
  tracking done using the unit fields,
  :class:`~eradiate.scenes.core.SceneElement` and its derived classes can still
  have their attributes initialised with :class:`pint.Quantity` instances (which
  *must* be created using Eradiate's unit registry
  :data:`eradiate.util.units.ureg`). When that happens,
  :class:`~eradiate.scenes.core.SceneElement`'s
  `post-init hook <https://www.attrs.org/en/stable/init.html#post-init-hook>`_
  will strip units from the attribute, after converting it to the stored unit.
  An important consequence is that this post-init hook must be executed by
  derived classes in order to retain this behaviour. Preferably, it should be
  the last to be executed.

  .. admonition:: Why unit fields?

     Eradiate's unit support uses unitless attributes and associated unit
     fields. Why this might seem overcomplicated, especially when comparing
     this workflow with direct use of :class:`pint.Quantity` objects, it
     allows for a very simple unit specification syntax when using dictionaries
     to initialise objects: units can be specified as strings in a JSON or YAML
     fragment.

     .. code-block:: python

        # This is the basic way and will store 1 m
        MyElement(field=1)
        # This will store 1 m
        MyElement(field=Quantity(1, ureg.m))
        # This will store 1 m
        MyElement(field=Quantity(100, ureg.cm), field_unit="m")
        # This will store 100 cm
        MyElement(field=100, field_unit=ureg.cm)
        # This will store 1 m (default unit is meter)
        MyElement.from_dict(yaml.load("""
            field: 1.
        """))
        # This will store 100 cm
        MyElement.from_dict(yaml.load("""
            field: 100.
            field_unit: cm
        """))

     The last example initialises the object correctly without the need of any
     YAML post-processing, which is something Eradiate takes advantage of.

* :class:`~eradiate.scenes.core.SceneElement` has a single abstract method
  :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` which must be
  implemented by its derived classes: it returns a dictionary which can be then
  used as an input of the Mitsuba kernel.

Constructing elements from the factory
--------------------------------------

The :class:`~eradiate.scenes.core.SceneElementFactory` class can be used to
construct registered :class:`~eradiate.scenes.core.SceneElement` derived classes.
Scene elements can be made accessible through Eradiate's factory system very
easily. The class definition simply has to be decorated using the
:meth:`SceneElementFactory.register() <eradiate.scenes.core.SceneElementFactory.register>`
decorator.

At this point, it is also important to check if the module in which the element
to be registered is located is properly registered as a search location in the
:class:`~eradiate.scenes.core.SceneElementFactory` class. By default,
:class:`~eradiate.scenes.core.SceneElementFactory` holds of list of submodules
where to search for factory-enabled classes; however, classes defined outside of
Eradiate's codebase won't be included in that list and it's the user's
responsibility to make sure that their custom element classes are imported at
some point so as to be registered to the factory.

In practice: Steps to write a new scene element class
-----------------------------------------------------

Following the above description, a new scene element class requires the following steps:

1. Derive a new class from :class:`~eradiate.scenes.core.SceneElement`. Decorate
   it with :func:`attr.s`.
2. Declare your custom attributes using :func:`~eradiate.util.attrs.attrib`.
   Don't hesitate to use the ``has_unit`` parameter to leverage the automatic
   unit handling system. If you do so, :func:`~eradiate.util.attrs.attrib_unit`
   will help you define your unit fields.
3. Implement the :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method.
   Things to keep in mind:

   * kernel imports must be local to the
     :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method;
   * the function's signature should allow for the processing of a ``ref``
     keyword argument (but using it is not required).

The following steps are optional:

* implement a post-init hook steps using the ``__attrs_post_init__()`` method
  (don't forget to call
  :meth:`SceneElement.__attrs_post_init__() <eradiate.scenes.core.SceneElement.__attrs_post_init__()>`
  at some point or you'll lose the unit handling);
* enable factory-based instantiation using the
  :meth:`SceneElementFactory.register() <eradiate.scenes.core.SceneElementFactory.register>` decorator.
