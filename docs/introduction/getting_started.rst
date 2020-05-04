.. _getting_started:

Getting Started
===============

.. warning::
   This page is outdated

Compiling
---------

Compiling from scratch requires CMake and a recent version of Xcode on macOS or Clang on Linux.

On Linux and macOS, compiling should be as simple as

.. code-block:: bash

    git clone --recursive git@europa.local:rtm/eradiate.git
    cd eradiate
    
    # Build using CMake & Ninja (recommended, must install Ninja
    # (https://ninja-build.org/) first)
    cmake . -B build -GNinja
    ninja


Running Eradiate
----------------

Once Eradiate is compiled, run the ``setpath.sh`` script to configure
environment variables (``PATH/LD_LIBRARY_PATH/PYTHONPATH``) that are
required to run Eradiate.

.. code-block:: bash

    source setpath.sh


Eradiate can then be used to compute radiative transfer in a scene by typing

.. code-block:: bash

    eradiate scene.xml

where ``scene.xml`` is a Eradiate scene file. Calling ``eradiate --help`` will print additional information about various command line arguments.

Running the tests
-----------------
To run the test suite, simply invoke ``pytest``:

.. code-block:: bash

    pytest

The build system also exposes a ``pytest`` target that executes ``setpath`` and
parallelizes the test execution.

.. code-block:: bash

    ninja pytest


Staying up-to-date
------------------

Eradiate organizes its software dependencies in a hierarchy of sub-repositories
using *git submodule*. Unfortunately, pulling from the main repository won't
automatically keep the sub-repositories in sync, which can lead to various
problems. The following command installs a git alias named ``pullall`` that
automates these two steps.

.. code-block:: bash

    git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

Afterwards, simply write

.. code-block:: bash

    git pullall

to stay in sync.

