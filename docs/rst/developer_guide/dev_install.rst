.. _sec-developer_guide-dev_install:

Development installation
========================

This guide covers all steps necessary to get Eradiate running on your machine in
development mode. This mode allows for easy code modifications and testing, as
well as for developing on the kernel code. It requires to compile the C++ code
of the Mitsuba renderer, using the Eradiate specific plugins and the appropriate
variants.

.. warning::

   Windows support is currently experimental. Please report issues on our
   `issue tracker <https://github.com/eradiate/eradiate/issues>`_.

Prerequisites
-------------

The development installation requires that the `Pixi <https://pixi.sh/>`_ package
manager is installed on the host machine. This tool replaces Conda and a lot of
in-house tools that were previously maintained by the development team.

Next, make sure that your machine meets the requirements listed below:

.. csv-table::
   :header: Requirement, Tested version
   :widths: 10, 10

   git,                     2.18+
   cmake,                   3.22+
   ninja,                   1.10+
   clang (macOS and Linux), 11+
   MSVC (Windows),          2022

.. dropdown:: Tested configuration
   :color: info
   :icon: info

   .. tab-set::
      .. tab-item:: Linux
         :sync: linux

         Operating system: Ubuntu Linux 22.04.1.

         .. csv-table::
            :header: Requirement, Tested version
            :widths: 10, 10

            git,       2.34.1
            cmake,     3.28.0
            ninja,     1.10.1
            clang,     11.1.0-6
            libc++,    11
            libc++abi, 11
            python,    3.9.18 (miniconda3)

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
            python, 3.9.18 (miniconda3)

      .. tab-item:: Windows
         :sync: windows

         Operating system: Windows 11.

         .. csv-table::
            :header: Requirement, Tested version
            :widths: 10, 20
            :stub-columns: 1

            git,    2.43.0.windows.1
            cmake,  3.22.1
            ninja,  1.10.2
            clang,  Apple clang version 13.0.0 (clang-1300.0.29.30)
            python, 3.9.18 (miniconda3)

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      All prerequisites except for conda can be installed through the usual
      Linux package managers. For example, using the APT package manager, which
      is used in most Debian-based distributions, like Ubuntu:

      .. code:: shell

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

      .. code:: shell

         brew install cmake ninja

      Additionally, running the Xcode command line tools once might be
      necessary:

      .. code:: shell

         xcode-select --install

   .. tab-item:: Windows
      :sync: windows

      On Windows, you will need to install the MSVC compiler, *e.g* through
      `Visual Studio Community <https://visualstudio.microsoft.com/>`_. In
      addition, you will need to install `GNU Make <https://gnuwin32.sourceforge.net/packages/make.htm>`_
      and `CMake <https://cmake.org/>`_.

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

      .. code:: shell

         git clone --recursive https://github.com/eradiate/eradiate

   .. tab-item:: Specific branch or tag

      .. code:: shell

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

Setting up the Python environment
---------------------------------

Pixi maintains a *lock file* that allows to quickly set up a reproducible
Python environment. To configure the develop environment, simply navigate to the
root of the cloned repository and install it:

.. code:: shell

   pixi install -e dev

Once installed, we **strongly recommend** to activate the Pixi environment in a
shell, either by spawning a Pixi shell

.. code:: shell

   pixi shell -e dev

or by activating the shell hook (discouraged by the Pixi documentation but
useful in specific cases)

.. code:: shell

   eval "$(pixi shell-hook -e dev)"

The `Pixi documentation <https://pixi.sh/latest/features/environment/#activation>`_
provides more detail about activation modes and their respective trade-offs. The
reason why we recommend this is that compiling the kernel often requires setting
environment variables, which is not always convenient outside of a shell. Note
that this is a recommendation, not a strict requirement: Pixi also allows to
set environment variables upon environment activation; we simply do not want to
encourage developers to modify their project manifest file for their particular
setup.

.. note::

   Although Pixi environments are very similar to Conda environment, there are
   significant differences. One of them is that the environment is not globally
   available for activation. If you want to activate your development
   environment from outside the project, this can be done with the
   ``--manifest-path`` option of the ``pixi shell`` command:

   .. code:: shell

      pixi shell --manifest-path /some/directory/pyproject.toml -e dev

   See the `Pixi CLI documentation <https://pixi.sh/latest/reference/cli/#shell>`_
   for details.

.. _sec-developer_guide-dev_install-compiling:

Compiling the radiometric kernel
--------------------------------

.. important::

   It is strongly recommended to activate the Pixi environment to compile the
   kernel. See :ref:`sec-developer_guide-dev_install-setup_conda` for details.

We recommend using the using the dedicated Pixi task to build the kernel:

.. code:: shell

   pixi run build-kernel

.. dropdown:: CMake Error: The source directory "..." does not exist
   :color: info
   :icon: info

   This most probably means that your CMake version is too old (see
   `Prerequisites`_). Up-to-date versions of CMake can be installed from many
   different sources: `pick you favourite <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_.

**Linux and macOS**: Inspect CMake's output to check if Clang is used as the C++
compiler. Search for lines starting with

.. code:: text

   -- The CXX compiler identification is <...>
   -- The C compiler identification is <...>
   -- Check for working CXX compiler: <...>
   -- Check for working C compiler: <...>

These lines should clearly indicate that the currently selected compiler is
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

         .. code:: shell

            export CC=clang-11
            export CXX=clang++-11

      .. tab-item:: macOS
         :sync: macos

         .. code:: shell

            export CC=clang
            export CXX=clang++

**All platforms**: Inspect CMake's output to check if your Conda environment
Python is used by CMake. Search for a line starting with:

.. code:: text

   -- Found Python: <...>

The content of this line may vary depending on the location of the project and
your Pixi configuration. If this path points to a Python binary not associated
with the target Pixi environment, but instead *e.g.* a global Python binary, do
not proceed before fixing it.

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

.. |smile| unicode:: U+1F642

In a terminal, try and invoke the :program:`eradiate` command-line interface:

.. code:: shell

   eradiate show

The command should print some information to the terminal. You are now ready to
use Eradiate |smile|
You probably also want to download part or all of Eradiate's built-in datasets:
see the :ref:`data guide <sec-data-intro-download>` for more information.

.. dropdown:: If you get a jit_cuda_compile() error ...
   :color: info
   :icon: info

   Eradiate does not use any CUDA variant of Mitsuba. You can therefore
   hide your graphics card by setting

   .. code:: shell

      export CUDA_VISIBLE_DEVICES=""

   Even doing so, you might still see a CUDA-related warning upon importing
   Eradiate. This is not a concern and it should be fixed in the future.
