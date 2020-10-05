.. _sec-getting_started-building:

Building Eradiate
=================

Before continuing, please make sure that you have read and followed the
instructions on :ref:`cloning Eradiate and its dependencies <sec-getting_started-getting_code>`.

It is highly recommended to use Eradiate in an isolated Python environment. Eradiate is developed using Conda to manage environments, available *e.g.* from the minimal Python distribution `Miniconda 3 <https://docs.conda.io/en/latest/miniconda.html>`_. However, nothing currently prevents Eradiate from running in a regular pyenv environment.

.. _sec-getting_started-building-python:

Setting up your Python environment
----------------------------------

.. note::

   For automated environment setup, please `see below <sec-getting_started-setup_automation>`.

.. _sec-getting_started-building-python-conda:

Using Conda [Recommended]
^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate requires a recent version of Python (at least **3.6**). A Conda environment file is provided in the ``resources/deps/`` directory and can be used to create a new environment:

.. code-block:: bash

    conda env create --file resources/deps/requirements_conda.yml --name eradiate

Once your environment is ready, you can activate it:

.. code-block:: bash

    conda activate eradiate

.. _sec-getting_started-building-python-conda-optional:

Optional requirements
"""""""""""""""""""""

The requirement file ``requirements_conda.yml`` contains all modules that are required to use Eradiate, but additional modules are available, which are used by developers.
The optional modules can be installed from the following files:

Developer requirements
    The file ``requirements_dev_conda.yml`` contains modules that are necessary for the development of Eradiate. This includes ``pytest`` and ``sphinx``, including extensions for them.
Jupyter lab extensions
    The files ``requirements_jupyter_conda.yml`` contains jupyter lab and extensions for it, which enable interactive usage of Eradiate in jupyter notebooks. The ``ipywidgets`` module enables proper rendering of ``tdqm's`` progress bars inside the jupyter notebook browser.

.. admonition:: Enabling jupyter extensions

    The jupyter extensions require two extra setup steps. These steps are necessary irrespective of the type of environment users employ.

    .. code-block:: bash

        jupyter nbextension enable --py widgetsnbextension
        jupyter labextension install @jupyter-widgets/jupyterlab-manager

.. _sec-getting_started-building-python-without_conda:

Installing without Conda
^^^^^^^^^^^^^^^^^^^^^^^^

We provide requirements files for use with pip, for the basic and developer requirements. These files can be found under ``resources/deps/requirements_pip.txt`` and
``resources/deps/requirements_dev_pip.txt``.

Additionally it is possible to directly :ref:`install the eradiate package <sec-getting_started-building-install_package>`. In this case the required packages will be installed through ``setup.py``.

.. _sec-getting_started-building-environment_variables:

Configuring environment variables
---------------------------------

Eradiate requires that a few environment variables (``PATH``/``PYTHONPATH``) are set. Run ``setpath.sh`` script to perform this setup:

.. code-block:: bash

    source setpath.sh

Note that this step is optional if you followed the instructions for :ref:`automated Conda environment setup <sec-getting_started-building-setup_automation>`


.. _sec-getting_started-building_mitsuba:

Building the Mitsuba kernel
---------------------------

Compiling Mitsuba 2 requires a recent version of CMake (at least **3.9.0**). Further platform-specific dependencies and compilation instructions are provided below for each operating system.

Linux prerequisites
^^^^^^^^^^^^^^^^^^^

The following table lists software that was installed on a fresh setup of Linux (Ubuntu 20.04.1).

 ============= =================
  Requirement   tested version
 ============= =================
  git           2.25.1
  cmake         3.16.3
  ninja         1.10.0
  Conda         4.8.3
  clang         10.0.0-4ubuntu1
  libc++        10
  libc++abi     10
  libpng
  zlib
  libjpeg
 ============= =================

.. admonition:: Installing packages

    All prerequisites except for conda can be installed through the usual Linux package managers. For example, using the APT package manager, which is used in most Debian based distributions, like Ubuntu:

    .. code-block:: bash

        # Install build tools, compiler and libc++
        sudo apt install -y git cmake ninja-build clang-10 libc++-dev libc++abi-dev

        # Install libraries for image I/O
        sudo apt install -y libpng-dev zlib1g-dev libjpeg-dev

    If your Linux distribution does not include APT, please consult your package manager's repositories for the respective packages.

Miniconda does not provide packages for the usual Linux package managers as of the writing of this document. However installers and installation instructions can be found on their `website <https://docs.conda.io/en/latest/miniconda.html>`_.

macOS prerequisites
^^^^^^^^^^^^^^^^^^^

On macOS, you will need to install Xcode, CMake, and `Ninja <https://ninja-build.org/>`_. Additionally, running the Xcode command line tools once might be necessary:

.. code-block:: bash

    xcode-select --install

.. admonition:: Tested configuration

    * macOS Catalina 10.15.2
    * Xcode 11.3.1
    * cmake 3.16.4
    * Python 3.7.3

Compiling
^^^^^^^^^

After following the steps for your OS above, compilation should be as simple as running the following from inside Eradiate's root directory:

.. code-block:: bash

    cd $ERADIATE_DIR
    mkdir build
    cd build
    cmake -GNinja ..
    ninja

Once Mitsuba is compiled, it can then be used to compute radiative transfer in a scene by typing

.. code-block:: bash

    mitsuba scene.xml

where ``scene.xml`` is a Mitsuba scene file. Calling ``mitsuba --help`` will print additional information about various command line arguments.

.. admonition:: Tips & Tricks

    Mitsuba compilation can fail due to CMake not accessing the correct Python interpreter and/or C/C++ compiler.
    In this case, the interpreter and compiler can be specified manually through CMake variables. To determine the path to the python interpreter run the following command in your terminal

    .. code-block:: bash

        which python

    The response should be a path, similar to this:

    .. code-block::

        /home/<username>/miniconda3/envs/eradiate/bin/python

    For the C and C++ compilers, run the following commands respectively.

    .. code-block:: bash

        which clang
        which clang++

    The resulting paths can be passed to CMake as variables, like this.

    .. code-block:: bash

        cmake -GNinja -D PYHTON_EXECUTABLE=<result of the query> CMAKE_C_COMPILER=<result of the query> CMAKE_CXX_COMPILER=<result of the query> ..

.. _sec-getting_started-building-install_package:

Installing Eradiate
-------------------

Once Mitsuba is compiled, Eradiate can be installed using pip:

.. code-block:: bash

    cd $ERADIATE_DIR
    pip install .

If you are modifying Eradiate's code, you should install it in editable mode:

.. code-block:: bash

    pip install -e .

Once this is done, you can check if the installation is successful by printing the embedded Mitsuba version number to the terminal:

.. code-block:: bash

    python -c "import eradiate.kernel; eradiate.kernel.set_variant('scalar_mono'); print(eradiate.kernel.core.MTS_VERSION)"

.. _sec-getting_started-building-setup_automation:

Setup automation
----------------

Conda environment creation can be automatically handled by executing the ``resources/envs/conda_create_env.sh`` script at the root of the Eradiate source tree. *Be careful however as this will reset the existing environment!*

The script automates by default the default Conda environment setup and the Eradiate package installation. Optional steps can also be automate using a series of flags:

* ``-d``: perform `development dependency installation <sec-getting_started-building-python-conda-optional>`
* ``-j``: perform `Jupyter lab installation and extension activation <sec-getting_started-building-python-conda-optional>`
* ``-a``: add automatic environment variable setup to environment activation script (makes the :ref:`environment variable setup <sec-getting_started-building-environment_variables>` no longer necessary)
* ``-e``: add a direnv ``.envrc`` file to the root of the Eradiate source tree (makes the :ref:`environment variable setup <sec-getting_started-building-environment_variables>` no longer necessary)

A typical convenient user setup can be done with the command:

.. code-block:: bash

    bash resources/envs/conda_create_env.sh -j -a

.. note::

   We are not sourcing the script, we are executing it in a subshell.

A typical developer setup can be done with the command:

.. code-block:: bash

    bash resources/envs/conda_create_env.sh -d -j -a
