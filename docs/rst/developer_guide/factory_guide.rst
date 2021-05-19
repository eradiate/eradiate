.. _sec-developer_guide-factory_guide:

Factory guide
=============

Eradiate's object creation process is heavily supported by a set of factory
classes. They serve two purposes:

* provide a safe and flexible converter system to support Eradiate's
  ``attrs``-based features;
* based on the previous mechanism, create objects from a uniform specification
  method based on (possibly nested) dictionaries.

This guide briefly introduces Eradiate's factories, how to use them and how to
document them.

Overview and usage
------------------

Eradiate's factories derive from the :class:`~eradiate._factory.BaseFactory`
class. This class can be used to create object factories which instantiate
objects based on dictionaries. Created factories handle only one type of object
(and child classes), specified in a ``_constructed_type`` class attribute. By
default, this member is set to :class:`object`: without any particular
precaution, a factory deriving from :class:`~eradiate._factory.BaseFactory`
accepts registration from any class, provided that it implements the required
interface.

.. note:: Factories deriving from :class:`~eradiate._factory.BaseFactory` are
   not meant to be instantiated and only implement class methods.

Class registration to factory objects is done using the
:meth:`~eradiate._factory.BaseFactory.register` class decorator (see
`Enabling a class for factory usage`_ below).

Using a factory created from this base class simply requires to import it and
call its :meth:`~eradiate._factory.BaseFactory.create` class method with a
dictionary containing:

- a ``type`` key, whose value will be the name under which the class to
  be instantiated is registered in the factory;
- dictionary contents which will be passed to the target class's
  ``from_dict()`` class method.

In addition, factories implement a :meth:`~eradiate._factory.BaseFactory.convert`
class method which can be used as a converter in an ``attrs`` initialisation
sequence. When passed an object, :meth:`~eradiate._factory.BaseFactory.convert`
forwards it to :meth:`~eradiate._factory.BaseFactory.create` if it is a
dictionary, and otherwise returns the passed object without doing anything.
When used as an ``attrs`` converter, this allows the user to initialise an
attribute by passing a dictionary interpreted by a factory instead of an object
of the requested. This enables the implementation of runtime instantiation and
configuration of classes from nested dictionaries, by reading them *e.g.* from
YAML or JSON fragments.

.. admonition:: Example

   The following code snippet instantiates a
   :class:`~eradiate.scenes.illumination.DirectionalIllumination` element
   using its :factorykey:`IlluminationFactory::directional` factory name:

   .. code:: python

       from eradiate.scenes.illumination import IlluminationFactory

       illumination = IlluminationFactory.create({
           "type": "directional",
           "irradiance": {"type": "uniform", "value": 1.0},
           "zenith": 30.0,
           "azimuth": 180.0,
       })

   In practice, the ``type`` key is used to look up the class to instantiate,
   then popped from the configuration dictionary. Therefore, the corresponding
   object creation call is, in this particular case:

   .. code:: python

       DirectionalIllumination(
           irradiance={"type": "uniform", "value": 1.0},
           zenith=30.0,
           azimuth=180.0,
       )

   Under the hood, this call creates a both
   :class:`~eradiate.scenes.illumination.DirectionalIllumination` and
   :class:`~eradiate.scenes.spectra.UniformSpectrum`: the former
   is instantiated directly (either implicitly using
   :meth:`.IlluminationFactory.create`, or explicitly using the
   :class:`~eradiate.scenes.illumination.DirectionalIllumination` constructor);
   the latter is instantiated when the dictionary passed as the ``irradiance``
   parameter is forwarded to :meth:`.SpectrumFactory.convert`.

Enabling a class for factory usage
----------------------------------

As previously mentioned, classes can be registered to a factory using the
factory's :meth:`~eradiate._factory.register` class decorator (which should
be applied *after* the :func:`attr.s` decorator). Decorated classes must
implement a ``from_dict()`` class method which generates instances from a
dictionary. If a class with an unsupported type is decorated with
:meth:`~eradiate._factory.BaseFactory.register`, a ``TypeError`` will be
raised upon import.

At this stage, factory features being implemented as class attributes and
methods is of prime importance. This means that factory registration is handled
at import time and allows for powerful and safe registration strategies. A
factory-enabled class must be imported so as to be registered to the factory.
When a factory and its classes are located in the same module, registration
is automatic (mind, however, that the declaration order is critical, so the
factory declaration must be placed before any call to its
:meth:`~eradiate._factory.BaseFactory.register` decorator).

.. note:: If a factory and classes to be registered to it are placed in
   different modules, importing the factory won't necessarily result in its
   register to be properly populated. For this reason, some factories can
   benefit from some additional registration code, which will make sure that
   modules containing classes to register will be discovered automatically when
   the factory will be imported. However, this can result in an unreliable
   import sequence, since discovery will inevitably introduce an import loop.
   For this reason, automatic module discovery is not performed by Eradiate's
   factories.

Documenting factories
---------------------

Printing a table of registered types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``.. factorytable::`` directive prints a table mapping a factory's keys to
the corresponding registered types:

.. tabbed:: ReST

   .. code-block:: restructuredtext

      .. factorytable::
         :factory: IlluminationFactory

.. tabbed:: Output

   .. factorytable::
        :factory: IlluminationFactory

This will create a factory key mapping table for the
:class:`eradiate.scenes.illumination.IlluminationFactory` class.

Referencing registered classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A registered class can be referenced by its factory key using the ``:factorykey:``
role.

.. tabbed:: ReST

   .. code-block:: restructuredtext

      The directional illumination scene element [:factorykey:`directional`] ...

.. tabbed:: Output

   The directional illumination scene element [:factorykey:`directional`] ...

This role takes a single argument, interpreted as the requested factory key.
If multiple factories use the same key to reference different types, the
referenced factory can be specified as a prefix, and using a ``::`` separator:

.. tabbed:: ReST

   .. code-block:: restructuredtext

      The directional illumination scene element
      [:factorykey:`IlluminationFactory::directional`] ...

.. tabbed:: Output

   The directional illumination scene element
   [:factorykey:`IlluminationFactory::directional`] ...


Advanced topics
---------------

Inheriting factory-registered types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The implementation chosen for factory registration makes it possible to safely
inherit factory-enabled types. They can themselves be registered to a factory,
possibly different from the one their parent class is registered to. The
following script shows an example:

.. literalinclude:: ../../examples/developer_guide/factory_guide/inheritance.py
   :language: python
