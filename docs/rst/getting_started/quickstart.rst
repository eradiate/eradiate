.. _sec-getting_started-quickstart:

Quickstart guide
================

This quickstart guide is an abridged version of the following two sections
:ref:`Getting the code <sec-getting_started-getting_code>` and :ref:`Building Eradiate <sec-getting_started-building>`.
It will cover all steps necessary to get Eradiate running on your machine, but it will not cover caveats and
common questions.

.. admonition:: Note

   If you experience issues with this quick start guide, please refer to the more detailed setup instructions.


.. _sec-getting_started-quickstart-prerequisites:

Prerequisites
-------------

Before cloning the repo and compiling the code, ensure that your machine meets the requirements
listed below.

.. tabbed:: Linux

   .. dropdown:: Tested configuration

      Operating system: Ubuntu Linux 20.04.1.

      .. csv-table::
         :header: Requirement, Tested version
         :widths: 10, 10
         :stub-columns: 1

         git,       2.25.1
         cmake,     3.16.3
         ninja,     1.10.0
         clang,     10.0.0-4ubuntu1
         libc++,    10
         libc++abi, 10

   .. admonition:: Installing packages

      All prerequisites except for conda can be installed through the usual Linux
      package managers. For example, using the APT package manager, which is used
      in most Debian based distributions, like Ubuntu:

      .. code-block:: bash

         # Install build tools, compiler and libc++
         sudo apt install -y git cmake ninja-build clang-10 libc++-dev libc++abi-dev

         # Install libraries for image I/O
         sudo apt install -y libpng-dev zlib1g-dev libjpeg-dev

      If your Linux distribution does not include APT, please consult your package
      manager's repositories for the respective packages.

.. tabbed:: macOS

   .. dropdown:: Tested configuration

      Operating system: macOS Catalina 10.15.2.

      .. csv-table::
         :header: Requirement, Tested version
         :widths: 10, 20
         :stub-columns: 1

         git,    2.24.2 (Apple Git-127)
         cmake,  3.18.4
         ninja,  1.10.1
         clang,  Apple clang version 11.0.3 (clang-1103.0.32.59)
         python, 3.7.9 (miniconda3)

   .. admonition:: Installing packages

      On macOS, you will need to install XCode, CMake, and
      `Ninja <https://ninja-build.org/>`_. XCode can be installed from the App
      Store. Make sure that your copy of the XCode is up-to-date. CMake and
      Ninja can be installed with the `Homebrew package manager <https://brew.sh/>`_:

      .. code-block:: bash

         brew install cmake ninja

      Additionally, running the Xcode command line tools once might be necessary:

      .. code-block:: bash

         xcode-select --install

Additionally Eradiate requires a fairly recent version of Python (at least 3.6) and we highly recommend
using the Conda environment manager to set up your Python environment.

.. _sec-getting_started-quickstart-cloning:

Cloning the repository
----------------------

To get the code, clone the repository including its submodules with the following command:

.. code-block:: bash

   git clone --recursive https://github.com/eradiate/eradiate

.. _sec-getting_started-quickstart-setup_conda:

Setting up the Conda environment
--------------------------------

Eradiate ships a shell script, which will set up a Conda environment with all
necessary packages and will add the required environment variables. Navigate to
the freshly created Git clone and run the script:

.. code-block:: bash

   cd eradiate
   bash resources/envs/conda_create_env.sh -j -a

Afterwards, activate the environment, running the following command

.. code-block:: bash

   conda activate eradiate

.. _sec-getting_started-quickstart-compiling:

Compiling the kernel
--------------------

Compilation is as simple as running the following from inside Eradiate's root directory:

.. code-block:: bash

   mkdir build
   cd build
   cmake -GNinja ..
   ninja

.. admonition:: Note

   If you activated the conda environment, the Eradiate root directory can be reached from everywhere
   through the ``$ERADIATE_DIR`` environment variable.


.. dropdown:: Tips & Tricks

   Mitsuba compilation can fail due to CMake not accessing the correct Python
   interpreter and/or C/C++ compiler.
   In this case, the interpreter and compiler can be specified manually through
   CMake variables. To determine the path to the python interpreter run the
   following command in your terminal

   .. code-block:: bash

      which python

   The response should be a path, similar to this:

   .. tabbed:: Linux

      .. code-block::

         /home/<username>/miniconda3/envs/eradiate/bin/python

   .. tabbed:: macOS

      .. code-block::

         /Users/<username>/miniconda3/envs/eradiate/bin/python

   For the C and C++ compilers, run the following commands respectively.

   .. code-block:: bash

      which clang
      which clang++

   The python interpreter is passed directly to cmake like this:

   .. code-block:: bash

      cmake -GNinja -D PYHTON_EXECUTABLE=<result of query> ..

   The C and C++ compilers must be defined through environment variables like this:

   .. code-block:: bash

      export CC=<result of query>
      export CXX=<result of query>

.. _sec-getting_started-quickstart-data_files:

Add large data files
--------------------

Download the `us76_u86_4-4000_25711 data set <https://eradiate.eu/data/spectra-us76_u86_4-fullrange.zip>`_,
extract the archive into a temporary location and copy contents into ``$ERADIATE_DIR/resources/data``.

Verify installation
-------------------

In a terminal, try and import Eradiate:

.. code-block:: bash

   python -c "import eradiate, eradiate.kernel; print(eradiate.__version__)"

The command should succeed and display the current version number.
You can now run Eradiate. |smile|

.. |smile| unicode:: U+1F642