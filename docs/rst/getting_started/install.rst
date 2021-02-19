.. _sec-getting_started-install:

Installation guide
==================

This guide covers all steps necessary to get Eradiate running on your machine.

.. warning:: Eradiate currently requires a development setup, even for end-
   users. This has a few consequences which we hope to overcome in the future:

   * installing and running Eradiate requires to use the Conda package and
     environment manager;
   * installation requires many dependencies not necessarily useful to
     end-users;
   * integrating Eradiate in your workflow can be a bit constraining if you have
     specific requirements on package versions.

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

      .. code:: bash

         # Install build tools, compiler and libc++
         sudo apt install -y git cmake ninja-build clang-9 libc++-9-dev libc++abi-9-dev

         # Install libraries for image I/O
         sudo apt install -y libpng-dev zlib1g-dev libjpeg-dev

      If your Linux distribution does not include APT, please consult your
      package manager's repositories for the respective packages.

   .. note:: We currently recommend compiling the C++ code with Clang based on
      `upstream advice from the Mitsuba development team <https://eradiate-kernel.readthedocs.io/en/latest/src/getting_started/compiling.html#linux>`_.
      We also recommend using Clang 9 — not another version — because we also
      encountered issues building with other versions. We hope to improve
      compiler support in the future.

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

      .. code:: bash

         brew install cmake ninja

      Additionally, running the Xcode command line tools once might be
      necessary:

      .. code:: bash

         xcode-select --install

Finally, Eradiate requires a fairly recent version of Python (at least 3.7)
and **we highly recommend using the Conda environment and package  manager** to
set up your Python environment. Conda can be installed notably as part of the
Anaconda distribution, or using its lightweight counterpart Miniconda.
`See installation instructions here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

.. _sec-getting_started-install-cloning:

Cloning the repository
----------------------

.. note::

   Eradiate relies on the `Git source code management tool <https://git-scm.com/>`_.
   It also depends on multiple external dependencies, some of which (*e.g.* its
   computational kernel based on
   `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_) are directly
   referred to using
   `Git submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.

To get the code, clone the repository including its submodules with the
following command:

.. code:: bash

   git clone --recursive https://github.com/eradiate/eradiate

This will clone the Eradiate repository, as well as all its dependencies. This
recursive cloning procedure can take up to a few minutes depending on your
Internet connection.

.. _sec-getting_started-install-setup_conda:

Setting up the Conda environment
--------------------------------

Eradiate ships a set of pinned Conda environment specifications in the form of
*lock files*. They quickly set up a reproducible environment. We strongly
recommend using these instead of a regular environment file since they provide
an execution environment identical to the one used for development.

In the following, we will use an environment named ``eradiate``, but this name
can be changed to your liking. We will first create an empty environment:

.. code:: bash

   conda create --name eradiate

.. warning:: If an environment with the same name exists, you will be prompted
   for overwrite.

This produces an empty environment, which we then activate:

.. code:: bash

   conda activate eradiate

We can now navigate to the repository where we cloned the source code and
execute a GNU Make target which will initialise our empty environment properly:

.. code:: bash

   cd eradiate
   make conda-init

.. admonition:: Notes

   * This target will not create a new Conda environment; it will instead
     install and/or update dependencies in the currently activated one.
   * This target will automatically select the appropriate lock file based
     on the platform on which you are working. It will also install Eradiate to
     your environment in development mode.
   * In addition to installing dependencies, this target will automate
     environment variable setup by sourcing ``setpath.sh`` upon environment
     activation, following
     `the approach recommended by the Conda user guide <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables>`_.
   * Once the Conda environment is active, the Eradiate root directory can
     be reached from everywhere through the ``$ERADIATE_DIR`` environment
     variable.

Once your Conda environment is configured, you should reactivate it:

.. code:: bash

   conda deactivate && conda activate eradiate

.. _sec-getting_started-install-compiling:

Compiling the kernel
--------------------

Create a build directory in Eradiate's root directory:

.. code:: bash

   mkdir build
   cd build

Configure CMake for compilation:

.. code:: bash

   cmake -GNinja -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..

Inspect CMake's output to check if Clang is used as the C++ compiler. Search for
lines starting with

.. code::

   -- Check for working C compiler: ...
   -- Check for working CXX compiler: ...

.. dropdown:: *If Clang is not used by CMake ...*

   If Clang is not used by CMake (this is very common on Linux systems), you
   have to explicitly define Clang as your C++ compiler. This can be achieved
   by modifying environment variables:

   .. code:: bash

      export CC=clang-9
      export CXX=clang++-9

   You might want to add these commands to your environment profile loading
   script. If you don't want to modify your environment variables, you can
   alternatively specify compilers during CMake configuration using CMake
   variables:

   .. code:: bash

      cmake -GNinja -DCMAKE_C_COMPILER=clang-9 -DCMAKE_CXX_COMPILER=clang++-9 -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..

Inspect CMake's output to check if your Conda environment Python is used by
CMake. Search for lines starting with:

.. tabbed:: Linux

      .. code::

         -- Found PythonInterp: /home/<username>/miniconda3/envs/eradiate/...
         -- Found PythonLibs: /home/<username>/miniconda3/envs/eradiate/...

.. tabbed:: macOS

   .. code::

      -- Found PythonInterp: /Users/<username>/miniconda3/envs/eradiate/...
      -- Found PythonLibs: /Users/<username>/miniconda3/envs/eradiate/...

.. dropdown:: *If the wrong Python binary is used by CMake ...*

   It probably means you have not activated your Conda environment:

   .. code:: bash

      conda activate eradiate

When CMake is successfully configured, you can compile the code:

.. code:: bash

   ninja

The compilation process can last for up to around half an hour on old machines.

.. _sec-getting_started-install-data_files:

Downloading required data files
-------------------------------

Eradiate does not automatically ship all available data sets due to their size.
In order to successfully run all tests and tutorials, the
`spectra-us76_u86_4 data set <https://eradiate.eu/data/spectra-us76_u86_4.zip>`_
must be downloaded manually and placed in the ``resources/data`` directory.
:ref:`This section <sec-user_guide-manual_download>` explains in detail where
the data set can be found and where it must be placed exactly.

Verifying the installation
--------------------------

In a terminal, try and import Eradiate:

.. code:: bash

   python -c "import eradiate.kernel; eradiate.kernel.set_variant('scalar_mono'); print(eradiate.kernel.core.MTS_VERSION)"

The command should succeed and display the current version number of the Mitsuba kernel.
You can now run Eradiate. |smile|

.. |smile| unicode:: U+1F642
