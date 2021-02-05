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

Before cloning the Git repository and compiling the code, ensure that your
machine meets the requirements listed below.

.. tabbed:: Linux

   .. dropdown:: *Tested configuration*

      Operating system: Ubuntu Linux 20.04.1.

      .. csv-table::
         :header: Requirement, Tested version
         :widths: 10, 10
         :stub-columns: 1

         git,       2.25.1
         cmake,     3.16.3
         ninja,     1.10.0
         clang,     9.0.1-12
         libc++,    9
         libc++abi, 9

   .. admonition:: Installing packages

      All prerequisites except for conda can be installed through the usual
      Linux package managers. For example, using the APT package manager, which
      is used in most Debian-based distributions, like Ubuntu:

      .. code-block:: bash

         # Install build tools, compiler and libc++
         sudo apt install -y git cmake ninja-build clang-9 libc++-9-dev libc++abi-9-dev

         # Install libraries for image I/O
         sudo apt install -y libpng-dev zlib1g-dev libjpeg-dev

      If your Linux distribution does not include APT, please consult your
      package manager's repositories for the respective packages.

.. tabbed:: macOS

   .. dropdown:: *Tested configuration*

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

      Additionally, running the Xcode command line tools once might be
      necessary:

      .. code-block:: bash

         xcode-select --install

Additionally Eradiate requires a fairly recent version of Python (at least 3.6)
and we highly recommend using the Conda environment manager to set up your
Python environment.

.. _sec-getting_started-quickstart-cloning:

Cloning the repository
----------------------

To get the code, clone the repository including its submodules with the
following command:

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

.. dropdown:: *Development setup*

   If you are setting up the code for development or want to run the test suite,
   then the ``-d`` flag will also add dev dependencies to the created Conda
   environment:

   .. code-block:: bash

      cd eradiate
      bash resources/envs/conda_create_env.sh -d -j -a

Afterwards, activate the environment, running the following command

.. code-block:: bash

   conda activate eradiate

.. admonition:: Note

   Once the Conda environment is active, the Eradiate root directory can be
   reached from everywhere through the ``$ERADIATE_DIR`` environment variable.

.. _sec-getting_started-quickstart-compiling:

Compiling the kernel
--------------------

Create a build directory in Eradiate's root directory:

.. code-block:: bash


   mkdir build
   cd build

Configure CMake for compilation:

.. code-block:: bash

   cmake -GNinja -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..

Inspect CMake output to check if clang is used as the C++ compiler. Search for
lines starting with

.. code-block::

   -- Check for working C compiler: ...
   -- Check for working CXX compiler: ...

.. dropdown:: *If clang is not used by CMake ...*

   If clang is not used by CMake (this is very common on Linux systems), you
   have to explicitly define clang as the default C++ compiler. This can be
   achieved with the following shell commands:

   .. code-block:: bash

      export CC=clang-9
      export CXX=clang++-9

   You might want to add these commands to your environment profile loading
   script.

Inspect CMake logs to check if your Conda environment Python is used by CMake.
Search for lines starting with:

.. tabbed:: Linux

      .. code-block::

         -- Found PythonInterp: /home/<username>/miniconda3/envs/eradiate/...
         -- Found PythonLibs: /home/<username>/miniconda3/envs/eradiate/...

.. tabbed:: macOS

   .. code-block::

      -- Found PythonInterp: /Users/<username>/miniconda3/envs/eradiate/...
      -- Found PythonLibs: /Users/<username>/miniconda3/envs/eradiate/...

.. dropdown:: *If the wrong Python binary is used by CMake ...*

   It probably means you have not activated your Conda environment:

   .. code-block:: bash

      conda activate eradiate

When CMake is successfully configured, you can compile the code:

.. code-block:: bash

   ninja

The compilation process can last for up to around half an hour on old machines.

.. _sec-getting_started-quickstart-data_files:

Adding large data files
-----------------------

Download the `us76_u86_4-4000_25711 data set <https://eradiate.eu/data/spectra-us76_u86_4-4000_25711.zip>`_,
extract the archive to a temporary location and copy contents into
``$ERADIATE_DIR/resources/data``.

Verifying the installation
--------------------------

In a terminal, try and import Eradiate:

.. code-block:: bash

   python -c "import eradiate.kernel; eradiate.kernel.set_variant('scalar_mono'); print(eradiate.kernel.core.MTS_VERSION)"

The command should succeed and display the current version number of the Mitsuba kernel.
You can now run Eradiate. |smile|

.. |smile| unicode:: U+1F642
