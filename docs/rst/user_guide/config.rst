.. _sec-user_guide-config:

Configuration
=============

Eradiate's configuration uses hierarchical settings. The settings configure
various parts of the behaviour of the library, ranging from data handling to
interpolation error handling. All settings have default values that can be
customized by users in several ways, with descending priority:

1. Environment variables.
2. An ``eradiate.toml`` file placed in the current working directory or higher.

Default values are :ref:`documented below <sec-user_guide-config-default>`.

.. warning::

   Although it is permitted, modifying the configuration at runtime is
   discouraged. Settings are assumed immutable during a session and changing
   them might result in untested behaviour.

In addition to settings, Eradiate has a few environment configuration items
that can only be set using environment variables:

:data:`ERADIATE_SOURCE_DIR <eradiate.config.SOURCE_DIR>`
    If set, this configuration item points to the directory where the local copy
    of the source code is cloned. In that case, Eradiate runs in development
    mode.

.. seealso::

   The full configuration reference is available in the :mod:`eradiate.config`
   reference documentation.

.. _sec-user_guide-config-default:

Default configuration file
--------------------------

.. literalinclude:: /resources/config/default.toml

