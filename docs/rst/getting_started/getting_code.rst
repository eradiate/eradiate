.. _sec-getting_started-getting_code:

Getting the code
================

Eradiate depends on multiple external dependencies, some of which (*e.g.* its computational kernel based on `Mitsuba 2 <https://github.com/mitsuba-renderer/mitsuba2>`_) are directly referred to using `Git submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_.

In order to clone Eradiate's Git repository and its submodules recursively, you have to use Git's ``--recursive`` flag:

.. code-block:: bash

    git clone --recursive git@europa.local:rtm/eradiate.git

If you already cloned the repository and forgot to specify this flag, it's possible to fix the repository in retrospect using the following command:

.. code-block:: bash

    git submodule update --init --recursive

Staying up-to-date
------------------

Unfortunately, pulling from the main repository won't automatically keep the submodules in sync, which can lead to various problems. The following command installs a git alias named ``pullall`` that automates these two steps.

.. code-block:: bash

    git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

Afterwards, simply write

.. code-block:: bash

    git pullall

to fetch the latest version of Eradiate and the appropriate versions of its nested submodules.
