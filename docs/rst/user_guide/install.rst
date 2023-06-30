.. _sec-user_guide-install:

Installation
============

Eradiate is delivered through PyPI and can be installed using the ``pip``. This
is the recommended way to install Eradiate.

.. code:: bash

   pip install eradiate


This will install the latest stable version of Eradiate, along with all the
dependencies necessary to run it. If you want to install the latest development
version, please refer to the :ref:`sec-developer_guide-dev_install`.

.. warning::

   Eradiate uses a modified version of the Mitsuba 3 renderer, distributed on
   PyPI as the ``eradiate-mitsuba``. That package conflicts with the ``mitsuba``
   package distributed by the Mitsuba team and both cannot be installed
   together.

   The ``eradiate`` PyPI package lists ``eradiate-mitsuba`` as a dependency. A
   normal usage pattern should result in the correct flavour of Mitsuba being
   installed automatically. However, if you are installing Eradiate to an
   environment already containing a Mitsuba installation, be sure to remove it
   before installing Eradiate.
