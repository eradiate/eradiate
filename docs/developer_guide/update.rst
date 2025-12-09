.. _sec-getting_started-update:

Update guide
============

Eradiate receives continuous updates. The use of submodules and locked Python
dependencies requires some care managing updates. This page provides guidelines
to help with this process.

Updating the source code
------------------------

In the cloned source directory, pull the latest update from GitHub:

.. code:: shell

   git pull

Unfortunately, pulling from the main repository won't automatically keep the
submodules in sync, which can lead to various problems. After pulling the
repository itself, it is essential to update the submodules. This is done using
the following command in the cloned source directory:

.. code:: shell

   git submodule update --init --recursive

.. dropdown:: Aliasing the update command for convenience
   :color: light
   :icon: info

   The following command installs a git alias named ``pullall`` that automates
   these two steps.

   .. code:: shell

      git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

   Afterwards, simply write

   .. code:: shell

      git pullall

   to fetch the latest version of Eradiate and the appropriate versions of its
   nested submodules.

The ``--init`` flag will ensure that any new submodule will be initialized.

Rebuilding the kernel
---------------------

After updating, it's always better to rebuild the kernel:

.. code:: shell

   pixi run build-kernel

Update data files
-----------------

Optionally, you may want to make sure that the data files you'll use are
up-to-date. To make sure that the data required for testing is correctly
installed, use the same procedure as for an end-user setup:

.. code:: shell

   eradiate data install core
