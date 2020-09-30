.. _sec-developer_guide-scene_helper_guide:

Writing a new scene helper
==========================

Scene helpers, deriving from the :class:`~eradiate.scenes.core.SceneHelper` class, are at the core of Eradiate's scene generation system. They provide an interface to quickly and safely generate kernel scene dictionary elements (see :class:`~eradiate.scenes.core.SceneDict` and :class:`~eradiate.scenes.core.KernelDict`).

The :class:`~eradiate.scenes.core.SceneHelper` base class
---------------------------------------------------------

:class:`~eradiate.scenes.core.SceneHelper` is the base class for all scene helpers. We will see here how this class works, and then how to write a new scene helper subclass.

Setting the configuration schema [:meth:`~eradiate.scenes.core.SceneHelper.config_schema`]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A critical component of scene helper classes is the *configuration schema*. Each scene helper must be configurable using a dictionary (they inherit the :class:`~eradiate.util.config_object.ConfigObject` class) and must therefore implement the :meth:`~eradiate.scenes.core.SceneHelper.config_schema` class method. The return Cerberus validation schema will be used upon initialisation to check whether the paramters passed to configure the helper are correct.

The default implementation of :meth:`~eradiate.scenes.core.SceneHelper.config_schema` simply checks that if the configuration dictionary passed to the constructor has an ``id`` field, it is a string.

Constructing a helper instance [:meth:`~eradiate.scenes.core.SceneHelper.init`]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next part we are interested in is the :class:`~eradiate.scenes.core.SceneHelper`'s constructor. It takes a single ``config`` argument, which is a dictionary. The ``config`` dictionary will be validated and normalised (*e.g.* to apply default parameter values) using the schema output by :meth:`~eradiate.scenes.core.SceneHelper.config_schema`.

If the helper requires additional construction steps, they can be easily implemented in the :meth:`~eradiate.scenes.core.SceneHelper.init` method. The default implementation does nothing.

Generating the kernel dictionary [:meth:`~eradiate.scenes.core.SceneHelper.kernel_dict`]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the helper is fully initialised, it can be used to produce a kernel dictionary section. This process is implemented by the :meth:`~eradiate.scenes.core.SceneHelper.kernel_dict`: it returns a dictionary which can be then used as an input of the Mitsuba kernel.

There is no default implementation for this method (it is what makes :class:`~eradiate.scenes.core.SceneHelper` an astract class).

Enabling factory support
^^^^^^^^^^^^^^^^^^^^^^^^

Scene helpers can be made accessible through Eradiate's factory system very easily. The class definition simply has to be decorated using the :meth:`Factory.register() <eradiate.scenes.core.Factory.register>` decorator.

At this point, it is also important to check if the module in which the helper to be registered is located is properly registered as a search location in the :class:`eradiate.scenes.core.Factory` class.

In practice: Steps to write a new scene helper class
----------------------------------------------------

Following the above description, a new scene helper class requires the following steps:

1. Derive a new class from :class:`~eradiate.scenes.core.SceneHelper`.
2. Implement the :meth:`~eradiate.scenes.core.SceneHelper.config_schema` class method (see the `Cerberus documentation <https://docs.python-cerberus.org/en/stable/index.html>`_ for an introduction to validation rules). Things to keep in mind:

   * :meth:`~eradiate.scenes.core.SceneHelper.config_schema` shall return a dictionary;
   * :meth:`~eradiate.scenes.core.SceneHelper.config_schema` should update the dictionary produced by its parent class, which can be retrieved using a ``super().config_schema()`` call;
   * fields can be associated another ``_unit`` field to leverage Eradiate's automated unit conversion system.

3. Implement the :meth:`~eradiate.scenes.core.SceneHelper.kernel_dict` method. Things to keep in mind:

   * kernel imports must be local to the :meth:`~eradiate.scenes.core.SceneHelper.kernel_dict` method;
   * the function's signature should allow for the processing of a ``ref`` keyword argument (but using it is not required).

The following steps are optional:

* implement additional constructor steps using the :meth:`~eradiate.scenes.core.SceneHelper.init` method;
* enable factory-based instatiation using the :meth:`Factory.register() <eradiate.scenes.core.Factory.register>` decorator.
