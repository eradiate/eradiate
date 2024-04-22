.. _sec-config:

``eradiate.config``
===================

.. automodule:: eradiate.config

Core members
------------

.. py:currentmodule:: eradiate.config

.. data:: ENV
   :annotation: = str

   Identifier of the environment in which Eradiate is used. Takes the value of
   the ``ERADIATE_ENV`` environment variable if it is set; otherwise defaults to
   ``"default"``.

.. data:: SOURCE_DIR
   :annotation: = pathlib.Path or None

   Path to the Eradiate source code directory, if relevant. Takes the value of
   the ``ERADIATE_SOURCE_DIR`` environment variable if it is set; otherwise
   defaults to ``None``.

.. py:currentmodule:: eradiate.config

.. data:: settings
   :annotation: = dynaconf.Dynaconf

   Main settings data structure. It supports array and attribute indexing, both
   case-insensitive:

   .. code:: python

      settings["some.key"]
      settings["SOME.KEY"]
      settings.some.key
      settings.SOME.KEY

   All settings have a default value specified by the
   :ref:`default configuration file <sec-user_guide-config-default>`
   and can be overridden by the user from an ``eradiate.toml`` file, placed in
   the current working directory or higher. Each setting can also be overridden
   using `environment variables <https://www.dynaconf.com/envvars/>`_ with the
   ``ERADIATE_`` prefix.

   .. admonition:: Example
      :class: tip

      The ``some.key`` setting will be accessed as ``ERADIATE_SOME__KEY`` (note
      the double underscore to figure the hierarchical separator).

Utility
-------

.. autoclass:: ProgressLevel
   :members:
