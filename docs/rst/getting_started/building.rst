.. _sec-getting_started-building:

Building Eradiate
=================

Before continuing, please make sure that you have read and followed the
instructions on :ref:`cloning Eradiate and its dependencies <sec-getting_started-getting_code>`.

It is highly recommended to use Eradiate in an isolated Python environment. Eradiate is developed using Conda to manage environments, available *e.g.* from the minimal Python distribution `Miniconda 3 <https://docs.conda.io/en/latest/miniconda.html>`_. However, nothing currently prevents Eradiate from running in a regular pyenv environment.

Setting up your Python environment
----------------------------------

Eradiate requires a recent version of Python (at least **3.6**). A Conda environment file is provided in the ``resources/environments`` directory and can be used to create a new environment:

.. code-block:: bash

    conda env create --file resources/environments/eradiate.yml --name eradiate_nested

Once your environment is ready, you can activate it:

.. code-block:: bash

    conda activate eradiate_nested

.. _sec-getting_started-automated_conda:

.. admonition:: Automated Conda environment setup

    Conda environment creation can be automatically handled by sourcing ``conda_create_env.sh`` script. *Be careful however as this will reset the existing environment!*

    .. code-block:: bash

        source conda_create_env.sh

    The created environment will also contain environment variable setup scripts which will make the :ref:`environment variable setup optional <sec-getting_started-environment_variables>`.

.. _sec-getting_started-environment_variables:

Configuring environment variables
---------------------------------

Eradiate requires that a few environment variables (``PATH``/``PYTHONPATH``) are set. Run ``setpath.sh`` script to perform this setup:

.. code-block:: bash

    source setpath.sh

Note that this step is optional if you followed the instructions for :ref:`automated Conda environment setup <sec-getting_started-automated_conda>`


Building the Mitsuba kernel
---------------------------

Compiling Mitsuba 2 requires a recent version of CMake (at least **3.9.0**). Further platform-specific dependencies and compilation instructions are provided below for each operating system.

Linux
~~~~~

.. warning::
    
    Add Linux installation instructions.

macOS
~~~~~

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

Installing Eradiate
-------------------

Once Mitsuba is compiled, Eradiate can be installed using the shipped setup script:

.. code-block:: bash

    python setup.py

If you are modifying Eradiate's code, you can install it in developer mode:

.. code-block:: bash

    python setup.py develop

Once this is done, you can check if the installation is successful by printing the embedded Mitsuba version to the terminal:

.. code-block:: bash

    python -c "import eradiate.kernel; print(eradiate.kernel.core.MTS_VERSION)"

Running the tests
-----------------

To run the test suite, invoke ``pytest`` with the following command:

.. code-block:: bash

    pytest eradiate

The Mitsuba test suite can also be run:

.. code-block:: bash

    pytest ext/mitsuba2/src
