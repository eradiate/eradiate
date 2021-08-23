.. _sec-developer_guide-factory_guide:

Factory guide
=============

Eradiate's object creation process is heavily supported by a set of instances of
the :class:`.Factory` class. They serve two purposes:

* provide a safe and flexible converter system to support Eradiate's
  ``attrs``-based features;
* based on the previous mechanism, create objects from a uniform specification
  method based on (possibly nested) dictionaries.

.. warning:: It is strongly advised to read the documentation of the
   `Dessine-moi library <https://dessinemoi.readthedocs.io/>`_ for an overview
   of Eradiate's factories.

Overview and usage
------------------

Eradiate's factories are instances of the :class:`.Factory` class, itself
derived from the :class:`dessinemoi.Factory` class.

Internally, a :class:`.Factory` instance maintains a *registry* which maps
string-typed identifiers to information required to create instances of
registered types. Each registry entry consists of the type registered under the
associated identifier, as well as an optional field specifying which class
method constructor should be used when converting dictionaries.

The :meth:`.Factory.convert` method is the main entry point to factories in
Eradiate. If this method is passed a dictionary, it pre-processes it, then tries
to instantiate one of the registered types based of the information contained in
the dictionary; otherwise, it does nothing and just returns the object it is
passed.

.. admonition:: Example

   The following code snippet instantiates a
   :class:`~eradiate.scenes.illumination.DirectionalIllumination` element
   using its ``directional`` factory identifier:

   .. code:: python

      from eradiate.scenes.illumination import illumination_factory

      illumination = illumination_factory.convert({
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

.. note::
   :meth:`.Factory.create` is powered by the low-level :meth:`.Factory.create`
   method.

Enabling a class for factory usage
----------------------------------

As previously mentioned, classes can be registered to a factory using the
factory's :meth:`.Factory.register` class decorator (which should
be applied *after* the :func:`attr.s` decorator). Our convention is to use the
``type_id`` keyword argument to declare the factory identifier—not a
``_TYPE_ID`` class attribute.

.. note::
   All the arguments of the :meth:`.Factory.register` decorator are keyword-only.

Documenting factories
---------------------

Documenting factories requires specific steps to work around Python's and
Sphinx's limitations regarding data member documentation. Upon adding a new
factory, please make sure you:

* add your new factory instance to the ``docs/generate_rst_api.py`` script
  (``FACTORIES`` variables);
* update the special API RST files (see also
  :ref:`sec-developer_guide-documentation-api-build`);
* add your new factory to the list of instances in the
  :ref:`sec-reference-factory` document (``docs/rst/reference/factory.rst``).
