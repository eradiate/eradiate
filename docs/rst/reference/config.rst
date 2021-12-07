.. _sec-config:

``eradiate._config``
====================

.. automodule:: eradiate._config

.. py:currentmodule:: eradiate._config

Classes
-------

.. autoclass:: EradiateConfig
   :members:

Attributes
----------

.. autodata:: config
   :annotation:

.. _sec-config-env_vars:

Environment variables
---------------------

.. exec::
   from eradiate._config import EradiateConfig, format_help_dicts_rst
   print(EradiateConfig.generate_help(formatter=format_help_dicts_rst, display_defaults=True))
