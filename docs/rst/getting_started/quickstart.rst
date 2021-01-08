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
      `Ninja <https://ninja-build.org/>`_. XCode can be install from the App
      Store. CMake and Ninja can be installed with the
      `Homebrew package manager <https://brew.sh/>`_:

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

Eradiate ships a shell script, which will set up a Conda environment with all necessary packages
and will add the required environment variables. Simply run the script like this:

.. code-block:: bash

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

.. _sec-getting_started-quickstart-install_package:

Installing the Eradiate package
-------------------------------

After successful compilation, you can install the Eradiate package in your conda environment.

.. code-block:: bash

   conda activate eradiate
   cd $ERADIATE_DIR
   pip install .

.. _sec-getting_started-quickstart-data_files:

Add large data files
--------------------

Download the ``us76_u86_4-4000_25711`` from `Link <https://eradiate.eu/data/spectra-us76_u86_4-4000_25711.zip>`_,
extract the archive into a temporary location and copy contents into ``$ERADIATE_DIR/resources/data``.

You can now run Eradiate. |smile|

.. |smile| unicode:: U+1F642