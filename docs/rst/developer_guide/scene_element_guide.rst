.. _sec-developer_guide-scene_element_guide:

Writing a new scene element class
=================================

.. warning:: Please first read carefully the
   :ref:`sec-developer_guide-conventions_practices` section. Writing scene
   elements requires general knowledge of the attrs, Pint and Pinttrs libraries.

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
  Although not required, this is a hint at the user: all scene element classes
  are written with the ``attrs`` library.
  :class:`~eradiate.scenes.core.SceneElement` has an ``id`` instance attribute
  with a default value: consequently, all instance attributes defined for
  child classes must have default values.

* :class:`~eradiate.scenes.core.SceneElement` derives from :class:`abc.ABC`: it
  is an abstract class and cannot be instantiated.

* :class:`~eradiate.scenes.core.SceneElement` has an abstract method
  :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` which must be
  implemented by its derived classes: it returns a dictionary which can be then
  used as an input to the kernel.

* :class:`~eradiate.scenes.core.SceneElement` has a class method constructor
  :meth:`~eradiate.scenes.core.SceneElement.from_dict` which implements a
  default dictionary interpretation protocol. This method can be overridden by
  derived classes.

* Unless specific action is taken, the default implementation of
  :meth:`.SceneElement.from_dict` requires that all :class:`.SceneElement`
  child classes only have keyword-argument attributes. It is therefore highly
  recommended to define defaults for all attributes, unless your new
  :class:`.SceneElement` provides an appropriate implementation of its
  :meth:`~.SceneElement.from_dict` method.

* :class:`~eradiate.scenes.core.SceneElement` is the base class of a set of
  abstract interfaces (*e.g.* :class:`~eradiate.scenes.illumination.Illumination`,
  :class:`~eradiate.scenes.spectra.Spectrum`, etc.) which are the ones from
  which new scene elements should derive.

Factory registration
--------------------

All interfaces derived from :class:`~eradiate.scenes.core.SceneElement` are
associated a specialised factory (see :ref:`sec-developer_guide-factory_guide`).
New :class:`~eradiate.scenes.core.SceneElement` subclasses should be registered
to the relevant factory so that Eradiate's dictionary-based object and
initialisation system works properly.

.. code-block:: python

   import attr
   from eradiate.scenes.spectra import Spectrum, SpectrumFactory
   from eradiate import ureg

   @SpectrumFactory.register("my_spectrum")
   @attr.s
   class MySpectrum(Spectrum):
       field = pinttr.ib(default=1.0, units=ureg.m)
       def eval(ctx=None): ...  # Definition skipped
       def kernel_dict(ctx=None): ...  # Definition skipped

   obj = SpectrumFactory.create({"type": "my_spectrum", "field": 1.0})

As mentioned in the :ref:`sec-developer_guide-factory_guide`, factory
registration occurs only upon class definition: a module defining a scene
element *must* be imported for the defined class to be registered to a factory.

Using factory converters
------------------------

As mentioned in the :ref:`sec-developer_guide-factory_guide`, Eradiate's
factories implement a :func:`~eradiate._factory.BaseFactory.convert` class
method which can turn a dictionary into a registered object—and if the method
receives something else than a dictionary, it simply does nothing.

This method can be used as a converter in the attribute initialisation sequence
to automatically convert a dictionary to a specified object. This allows for
the use of nested dictionaries to instantiate multiple objects.

.. code-block:: python

   import attr
   import pinttr

   from eradiate import unit_registry as ureg
   from eradiate.scenes.illumination import Illumination, IlluminationFactory
   from eradiate.scenes.spectrum import Spectrum, SpectrumFactory

   @SpectrumFactory.register("my_spectrum")
   @attr.s
   class MySpectrum(Spectrum):
       field = pinttr.ib(default=1.0, units=ureg.m)
       def eval(ctx=None): ...  # Definition skipped
       def kernel_dict(ctx=None): ...  # Definition skipped

   @IlluminationFactory.register("my_illumination")
   @attr.s
   class MyIllumination(Illumination):
       radiance = attr.ib(
           factory=MySpectrum,
           converter=SpectrumFactory.convert
       )
       def kernel_dict(): ...  # Definition skipped

   # Pass object created with constructor
   obj = MyIllumination(radiance=MySpectrum(field=2.0))
   # Use the factory to convert a dictionary to ElementA
   obj = MyIllumination(element_a={"type": "my_spectrum", "field": 3.0})
   # Instantiate MyIllumination using nested dicts
   obj = IlluminationFactory.create({
       "type": "my_illumination",
       "radiance": {"type": "my_spectrum", "field": 4.0},
   })

The :meth:`~.SceneElement.kernel_dict` method
---------------------------------------------

Any scene element **must** implement a :meth:`~.SceneElement.kernel_dict` method
which will return a dictionary suitable for merge into a kernel scene
dictionary. These dictionaries are written following the Mitsuba scene
specification and the interested reader is referred to kernel docs for further
information.

.. note:: When writing the :meth:`~.SceneElement.kernel_dict` method, there are
   a few precautions to keep in mind:

   * kernel imports must be local to the method;
   * if a  kernel importis required to build the dictionary, a kernel variant
     must be selected when it is called (in practice, this means that Eradiate's
     operational mode must have been selected);
   * :meth:`~.SceneElement.kernel_dict`'s signature should allow for the
     processing of a :class:`.KernelDictContext` instance, which carries around
     state variables during recursive kernel dictionary generation.

In practice: Steps to write a new scene element class
-----------------------------------------------------

Following the above description, a new scene element class requires the
following steps:

1. Derive a new class from one of the :class:`~eradiate.scenes.core.SceneElement`
   subclasses. Decorate it with :func:`attr.s`.
2. Declare your custom attributes using :func:`attr.ib`. Don't forget to add
   default values to all of them. Use :func:`pinttr.ib` if the field represents
   a physical quantity with units. Callables can be used to evaluate units
   dynamically. If the field requires it, it is possible to run custom
   converters and validators.
3. Implement the :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method.
   Things to keep in mind:

   * kernel imports must be local to the
     :meth:`~eradiate.scenes.core.SceneElement.kernel_dict` method;
   * the function's signature should allow for the processing of a ``ctx``
     keyword argument of type :class:`.KernelDictContext` (but using it is not
     required).

The following steps are optional:

* implement a post-init hook steps using the ``__attrs_post_init__()`` method;
* enable factory-based instantiation using the
  :meth:`~eradiate._factory.BaseFactory.register()` decorator defined by the
  appropriate factory.
