.. _sec-user_guide-install:

Installation
============

.. warning::

   Windows support is currently experimental. Please report issues on our
   `issue tracker <https://github.com/eradiate/eradiate/issues>`__.

Eradiate is delivered through PyPI and can be installed using the ``pip``
command:

.. code:: bash

   pip install 'eradiate[kernel]'

This will install the latest stable version of Eradiate, along with all the
dependencies necessary to run it. If you want to install the latest development
version, please refer to the :ref:`sec-developer_guide-dev_install`.

.. warning::

   Eradiate uses a modified version of the Mitsuba 3 renderer, distributed on
   PyPI as ``eradiate-mitsuba``. That package conflicts with the ``mitsuba``
   package distributed by the Mitsuba team. Both are incompatible with each
   other.

   The ``eradiate`` PyPI package lists ``eradiate-mitsuba`` as a dependency. A
   normal usage pattern should result in the correct flavour of Mitsuba being
   installed automatically. However, if you are installing Eradiate to an
   environment already containing a Mitsuba installation, be sure to remove it
   before installing Eradiate.

After installing Eradiate, it is recommended to download some support data to
start simulations:

.. code:: bash

   eradiate data install core
