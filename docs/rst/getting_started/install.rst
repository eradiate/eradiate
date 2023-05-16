.. _sec-getting_started-install:

Installation through Pypi
=========================

Eradiate is delivered through Pypi and can be installed using the ``pip``.

.. code:: bash

   pip install "eradiate[production]"


This will install the latest stable version of Eradiate, along with all the
dependencies necessary to run it. If you want to install the latest development,
please refer to the :ref:`sec-getting_started-development`.

.. warning::

   It is absolutely necessary to add the ``[production]`` option to the ``pip``
   command. This will ensure that the ``eradiate`` package is installed with
   the mitsuba backend, not installed by default. Some refactoring work is in
   progress to make this option unnecessary in the future.

