.. _sec-getting_started-building:

Building Eradiate
=================

Before continuing, please make sure that you have read and followed the
instructions on :ref:`cloning Eradiate and its dependencies <sec-getting_started-getting_code>`.

It is highly recommended to use Eradiate in an isolated Python environment. Eradiate is developed using Conda to manage environments, available *e.g.* from the minimal Python distribution `Miniconda 3 <https://docs.conda.io/en/latest/miniconda.html>`_. However, nothing currently prevents Eradiate from running in a regular pyenv environment.

Setting up your Python environment
----------------------------------

Using Conda [Recommended]
^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate requires a recent version of Python (at least **3.6**). A Conda environment file is provided in the ``resources/deps/`` directory and can be used to create a new environment:

.. code-block:: bash

    conda env create --file resources/deps/requirements_conda.yml --name eradiate

Once your environment is ready, you can activate it:

.. code-block:: bash

    conda activate eradiate

.. _sec-getting_started-building-automated_conda:

.. admonition:: Automated Conda environment setup

    Conda environment creation can be automatically handled by executing the ``conda_create_env.sh`` script. *Be careful however as this will reset the existing environment!* Note that we are not sourcing the script, we are executing it in a subshell.

    .. code-block:: bash

        bash conda_create_env.sh

    The created environment will also contain environment variable setup scripts which will make the :ref:`environment variable setup optional <sec-getting_started-building-environment_variables>`.

Optional requirements
"""""""""""""""""""""

The requirement file ``requirements_conda.yml`` contains all modules that are required to use Eradiate, but additional modules are available, which are used by developers.
The optional modules can be installed from the following files:

Developer requirements
""""""""""""""""""""""

The file ``requirements_dev_conda.yml`` contains modules that are necessary for the development of Eradiate. This includes ``pytest`` and ``sphinx``, including extensions for them.

Jupyter lab extensions
""""""""""""""""""""""

The files ``requirements_jupyter_conda.yml`` contains jupyter lab and extensions for it, which enable interactive usage of Eradiate in jupyter notebooks. The ``ipywidgets`` module
enables proper rendering of ``tdqm's`` progress bars inside the jupyter notebook browser.

.. admonition:: Enabling jupyter extensions

    The jupyter extensions require two extra setup steps. These steps are necessary irrespective of the type of environment users employ.

    .. code-block:: bash

        jupyter nbextension enable --py widgetsnbextension
        jupyter labextension install @jupyter-widgets/jupyterlab-manager

Installing without Conda
^^^^^^^^^^^^^^^^^^^^^^^^

We provide requirements files for use with pip, for the basic and developer requirements. These files can be found under ``resources/deps/requirements_pip.txt`` and
``resources/deps/requirements_dev_pip.txt``.

Additionally it is possible to directly :ref:`install the eradiate package <sec-getting_started-building-package_install>`. In this case the required packages will be installed through ``setup.py``.

.. _sec-getting_started-building-environment_variables:

Configuring environment variables
---------------------------------

Eradiate requires that a few environment variables (``PATH``/``PYTHONPATH``) are set. Run ``setpath.sh`` script to perform this setup:

.. code-block:: bash

    source setpath.sh

Note that this step is optional if you followed the instructions for :ref:`automated Conda environment setup <sec-getting_started-building-automated_conda>`


.. _sec-getting_started-building-mitsuba:

Building the Mitsuba kernel
---------------------------

Compiling Mitsuba 2 requires a recent version of CMake (at least **3.9.0**). Further platform-specific dependencies and compilation instructions are provided below for each operating system.

Linux
^^^^^

.. todo::
    
    Add Linux installation instructions.

macOS
^^^^^

On macOS, you will need to install Xcode, CMake, and `Ninja <https://ninja-build.org/>`_. Additionally, running the Xcode command line tools once might be necessary:

.. code-block:: bash

    xcode-select --install

.. admonition:: Tested configuration

    * macOS Catalina 10.15.2
    * Xcode 11.3.1
    * cmake 3.16.4
    * Python 3.7.3

Now, compilation should be as simple as running the following from inside Eradiate's root directory:

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

.. _sec-getting_started-building-package_install:

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

Setup automation
----------------

Conda environment creation can be automatically handled by executing the ``resources/envs/conda_create_env.sh`` script. *Be careful however as this will reset the existing environment!*

.. code-block:: bash

    bash resources/envs/conda_create_env.sh

Note that we are not sourcing the script, we are executing it in a subshell.

The script will install all optional dependencies for developers, jupyter lab and its required extensions and the Eradiate package in development mode. The created environment will also contain environment variable setup scripts which will make the :ref:`environment variable setup optional <sec-getting_started-building-environment_variables>`.
