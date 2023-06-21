.. _sec-getting_started-update:

Update guide
============

Eradiate receives continuous updates. The use of submodules and locked Python
dependencies requires some care managing updates. This page provides guidelines
to help with this process.

Updating the source code
------------------------

In the cloned source directory, pull the latest update from GitHub:

.. code:: bash

   cd $ERADIATE_SOURCE_DIR
   git pull

Unfortunately, pulling from the main repository won't automatically keep the
submodules in sync, which can lead to various problems. After pulling the
repository itself, it is essential to update the submodules. This is done using
the following command in the cloned source directory:

.. code:: bash

   git submodule update --init --recursive

.. dropdown:: Aliasing the update command for convenience
   :color: light
   :icon: info

   The following command installs a git alias named ``pullall`` that automates
   these two steps.

   .. code:: bash

      git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

   Afterwards, simply write

   .. code:: bash

      git pullall

   to fetch the latest version of Eradiate and the appropriate versions of its
   nested submodules.

The ``--init`` flag will ensure that any new submodule will be initialised.

Rebuilding the kernel
---------------------

After updating, it's always better to rebuild the kernel:

.. code:: bash

   cd $ERADIATE_SOURCE_DIR
   cmake --build build

Updating your Conda environment
-------------------------------

After updating the source code, an update of your Conda environment might be
necessary. In that case, the ``conda-init`` target can be used:

.. code:: bash

   cd $ERADIATE_SOURCE_DIR
   make conda-init

If something goes wrong during that process, an environment reset should solve
most issues (see :ref:`sec-developer_guide-dev_install-setup_conda`).

Update data files
-----------------

Optionally, you may want to make sure that the data files you'll use are
up-to-date. For that purpose, you can refresh remote file registries and purge
Eradiate's data cache:

.. code:: bash

   eradiate data update-registries
   eradiate data purge-cache --keep

.. note::
   For a more aggressive cleanup, just run

   .. code:: bash

      eradiate data purge-cache
