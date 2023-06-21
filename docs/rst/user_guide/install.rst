.. _sec-user_guide-install:

Installation
============

Eradiate is delivered through PyPI and can be installed using the ``pip``. This
is the recommended way to install Eradiate.

.. code:: bash

   pip install eradiate


This will install the latest stable version of Eradiate, along with all the
dependencies necessary to run it. If you want to install the latest development,
please refer to the :ref:`sec-developer_guides-dev_install`.

.. warning::

   The ``eradiate`` depends on the ``eradiate-mitsuba`` package. This latter is
   strictly incompatible with ``mitsuba`` package. If you have installed
   ``mitsuba`` before, please make sure to uninstall it before installing
   ``eradiate`` and/or ``eradiate-mitsuba``.

   The original ``mitsuba`` package provided by the Mitsuba team
   does not include the appropriate variants and addtional modules for Eradiate
   to function properly. Please make sure to install the ``eradiate`` package
   with its ``eradiate-mitsuba`` dependency.
