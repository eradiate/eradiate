.. _sec-developer_guide-dev_install:

Development installation
========================

This guide covers all steps necessary to get Eradiate running on your machine in
development mode. This mode allows for easy code modifications and testing, as
well as for developing on the kernel code. It requires to compile the C++ code
of the Mitsuba renderer, using the Eradiate specific plugins and the appropriate
variants.

Prerequisites
-------------

Before cloning the Git repository and compiling the code, ensure that your
machine meets the requirements listed below.

.. csv-table::
   :header: Requirement, Tested version
   :widths: 10, 10

   git,       2.18+
   cmake,     3.22+
   ninja,     1.10+
   clang,     11+

.. dropdown:: Tested configuration
   :color: info
   :icon: info

   .. tab-set::
      .. tab-item:: Linux
         :sync: linux

         Operating system: Ubuntu Linux 20.04.1.

         .. csv-table::
            :header: Requirement, Tested version
            :widths: 10, 10

            git,       2.25.1
            cmake,     3.22.2
            ninja,     1.10.0
            clang,     11.0.0-2
            libc++,    11
            libc++abi, 11
            python,    3.8.12 (miniconda3)

      .. tab-item:: macOS
         :sync: macos

         Operating system: macOS Monterey 12.0.1.

         .. csv-table::
            :header: Requirement, Tested version
            :widths: 10, 20
            :stub-columns: 1

            git,    2.32.0 (Apple Git-132)
            cmake,  3.22.1
            ninja,  1.10.2
            clang,  Apple clang version 13.0.0 (clang-1300.0.29.30)
            python, 3.8.12 (miniconda3)

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      All prerequisites except for conda can be installed through the usual
      Linux package managers. For example, using the APT package manager, which
      is used in most Debian-based distributions, like Ubuntu:

      .. code:: bash

         # Install build tools, compiler and libc++
         sudo apt install -y git cmake ninja-build clang-11 libc++-11-dev libc++abi-11-dev

         # Install libraries for image I/O
         sudo apt install -y libpng-dev zlib1g-dev libjpeg-dev

      If your Linux distribution does not include APT, please consult your
      package manager's repositories for the respective packages.

      If your CMake copy is not recent enough, there are
      `many ways <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_
      to install an updated version, notably through pipx and Conda. Pick your
      favourite!

      .. note:: We currently recommend compiling the C++ code with Clang based on
         `upstream advice from the Mitsuba development team <https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html#linux>`_.
         We also recommend using Clang 11 — not another version — because we also
         encountered issues building with other versions. We hope to improve
         compiler support in the future.

   .. tab-item:: macOS
      :sync: macos

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

Finally, Eradiate requires a fairly recent version of Python (at least 3.8)
and **we highly recommend using the Conda environment and package  manager** to
set up your Python environment. Conda can be installed notably as part of the
Anaconda distribution, or using its lightweight counterpart Miniconda.
`See installation instructions here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

.. _sec-developer_guide-dev_install-cloning:

Cloning the repository
----------------------

.. note::

   Eradiate relies on the `Git source code management tool <https://git-scm.com/>`_.
   It also depends on multiple external dependencies, some of which (*e.g.* its
   radiometric kernel based on
   `Mitsuba 3 <https://github.com/mitsuba-renderer/mitsuba3>`_) are directly
   referred to using
   `Git submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.

To get the code, clone the repository including its submodules with the
following command:

.. tab-set::

   .. tab-item:: Latest main branch

      .. code:: bash

         git clone --recursive https://github.com/eradiate/eradiate

   .. tab-item:: Specific branch or tag

      .. code:: bash

         git clone --recursive --branch <ref> https://github.com/eradiate/eradiate

      where ``<ref>`` is a Git branch or tag. For the latest stable version, use
      ``stable``.

This will clone the Eradiate repository, as well as all its dependencies.
This recursive cloning procedure can take up to a few minutes depending on
your Internet connection.

.. note::

   If GitHub requests credentials to access submodules through HTTPS, we highly
   recommend to `generate a personal access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_
   with **repo** permissions and to use it instead of your password. You might
   also have to make sure that `Git will remember your token <https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage>`_.

.. _sec-developer_guide-dev_install-setup_conda:

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

.. warning::
   If an environment with the same name exists, you will be prompted for
   overwrite.

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
     be reached from everywhere through the ``$ERADIATE_SOURCE_DIR`` environment
     variable.

Once your Conda environment is configured, you should reactivate it:

.. code:: bash

   conda deactivate && conda activate eradiate

.. _sec-developer_guide-dev_install-compiling:

Compiling the radiometric kernel
--------------------------------

Using the Makefile rule to build the kernel is the recommended way to compile.

.. code:: bash

   make kernel

.. dropdown:: CMake Error: The source directory "..." does not exist
   :color: info
   :icon: info

   This most probably means that your CMake version is too old
   (see `Prerequisites`_). At this stage, you might also install CMake in your
   Conda environment:

   .. code:: bash

      conda install "cmake>=3.22"

Inspect CMake's output to check if Clang is used as the C++ compiler. Search for
lines starting with

.. code::

   -- Check for working C compiler: ...
   -- Check for working CXX compiler: ...

If you see ``gcc`` on this line, it very likely means that CMake is not using
Clang.

.. dropdown:: If Clang is not used by CMake ...
   :color: info
   :icon: info

   If Clang is not used by CMake (this is very common on Linux systems, less
   likely on macOS), you have to explicitly define Clang as your C++ compiler.
   This can be achieved by modifying environment variables:

   .. tab-set::

      .. tab-item:: Linux
         :sync: linux

         .. code:: bash

            export CC=clang-11
            export CXX=clang++-11

      .. tab-item:: macOS
         :sync: macos

         .. code:: bash

            export CC=clang
            export CXX=clang++


Inspect CMake's output to check if your Conda environment Python is used by
CMake. Search for a line starting with:

.. tab-set::

      .. tab-item:: Linux
         :sync: linux

         .. code::

            -- Found Python: /home/<username>/miniconda3/envs/eradiate/...

      .. tab-item:: macOS
         :sync: macos

         .. code::

            -- Found Python: /Users/<username>/miniconda3/envs/eradiate/...

The content of this line may vary depending on how you installed Conda. If
this path points to a Python binary not associated with a Conda virtual
environment, do not proceed before fixing it.

.. dropdown:: If the wrong Python binary is used by CMake ...
   :color: info
   :icon: info

   It probably means you have not activated your Conda environment:

   .. code:: bash

      conda activate eradiate

.. note::

   You will probably see a warning saying

       *Created a default 'mitsuba.conf' configuration file.  You will
       probably want to edit this file to specify the desired configurations
       before starting to compile.*

   This is expected: do not worry about it.

The compilation process can last for up to around half an hour on old machines.
It completes within a few minutes on modern workstations.

.. _sec-developer_guide-dev_install-verify_installation:

Verifying the installation
--------------------------

In a terminal, try and invoke the :program:`eradiate` command-line interface:

.. code:: bash

   eradiate show

The command should print some information to the terminal. You are now ready to
use Eradiate |smile|

.. |smile| unicode:: U+1F642

.. dropdown:: If you get a jit_cuda_compile() error ...
   :color: info
   :icon: info

   Eradiate does not use any CUDA variant of Mitsuba. You can therefore
   hide your graphics card by setting

   .. code:: bash

      export CUDA_VISIBLE_DEVICES=""

   Even doing so, you might still see a CUDA-related warning upon importing
   Eradiate. This is not a concern and it should be fixed in the future.

Uninstall
---------

To uninstall Eradiate from your system, simply remove the Conda environment you
used to set it up and delete the directory where you cloned the code. If you
followed the installation instructions, here is a possible workflow:

1. Activate the Conda environment and delete the Eradiate source directory:

   .. code:: bash

      conda activate eradiate
      rm -rf $ERADIATE_SOURCE_DIR

2. Deactivate the Conda environment and delete it:

   .. code:: bash

      conda deactivate
      conda env remove --name eradiate
