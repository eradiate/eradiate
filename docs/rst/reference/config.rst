.. _sec-config:

Configuration [eradiate._config]
================================

.. currentmodule:: eradiate._config

.. autosummary::
   :toctree: generated/

   EradiateConfig

.. _sec-config-env_vars:

Environment variables
---------------------

.. exec::
   from eradiate._config import EradiateConfig, format_help_dicts_rst
   print(EradiateConfig.generate_help(formatter=format_help_dicts_rst))
