.. _sec-maintainer_guide:

Maintainer guide
================

.. _sec-maintainer_guide-dependencies:

Managing dependencies
---------------------

.. _Pixi: https://pixi.sh/
.. _Pixi Basic usage: https://pixi.sh/latest/basic_usage/
.. _Pixi Python development: https://pixi.sh/latest/tutorials/python/

Eradiate is managed using the `Pixi`_ project manager. It notably allows
us to maintain a Conda-based setup with most dependencies sourced from PyPI,
with layered requirements. Be sure to read the relevant entries in the Pixi
documentation:

* `Pixi Basic usage`_
* `Pixi Python development`_

We use the following requirement groups, which manifest as `features` in the
Pixi model:

* ``optional``: Only contains the Eradiate kernel package. This is needed to
  allow developers to install all dependencies except the kernel.
* ``recommended``: Optional packages used by specific subcomponents of Eradiate.
* ``docs``: Packages needed to compile the documentation.
* ``test``: Packages needed to run the test suite.

When adding new requirements, be sure to:

* Prioritize PyPI packages, using the ``--pypi`` option.
* Register the new requirement to the appropriate group.
* Check in the lock file after it is updated.

.. note::

   Our Read The Docs build uses stock Python virtual environments and installs
   dependencies with Pip. A specific requirement file is generated for that
   purpose, using the ``docs-lock`` Pixi task. This task is automatically
   executed upon committing, so the file is always up-to-date.

.. _sec-maintainer_guide-release:

Making a release of Eradiate
----------------------------

.. _start a Pixi shell: https://pixi.sh/latest/features/environment/#activation
.. _Bump My Version: https://github.com/callowayproject/bump-my-version

1. **Preparation**

   1. Make sure main is up-to-date and all tests pass.
   2. If necessary, [1]_ `start a Pixi shell`_:

      .. code:: shell

         pixi shell -e dev

   3. In your shell, set the variable ``RELEASE_VERSION`` to the target version
      value:

      .. code:: shell

         export RELEASE_VERSION=X.Y.Z

   4. Create a new branch for the release:

      .. code:: shell

         git checkout main && git pull upstream main && git checkout -b bump/prepare-v$RELEASE_VERSION

   5. Make sure that dependencies are correct (check in particular the kernel
      version). Use the release checker utility for this:

      .. code:: shell

         pixi run release check-mitsuba

   6. Bump the version number using `Bump My Version`_:

      .. code:: shell

         pixi run bump

   7. Update the change log.
   8. Commit the changes:

      .. code:: shell

         git commit -am "Bump version to $RELEASE_VERSION"

   9. Update the version and release date fields in ``CITATION.cff``:

      .. code:: shell

         pixi run release update-citation

   10. Update the PyPI README content:

       .. code:: shell

          pixi run release update-pypi-readme

   11. Push the changes:

       .. code:: shell

          git push origin

2. **Pull request**

   1. Create a pull request to check changes with peers.
   2. Merge the pull request once everything is correct.

3. **Release publication**

   1. Create a draft release on GitHub and update it.
   2. Using release candidates on Test PyPI, make sure that built Pyhon wheels
      will work as expected. A typical installation command of a Test PyPI
      release is

      .. code:: shell

         python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'eradiate[kernel]==0.30.0rc4'

   3. Finalize release notes and create the release tag. **Make sure that the
      release commit is referenced only by one tag.**
   4. Build and upload Python wheels.

4. **Post-release: Prepare the next development cycle**

   1. In your shell, set the variable ``RELEASE_VERSION`` to the target version
      value:

      .. code:: shell

         export RELEASE_VERSION=X.Y.Z-dev0

   2. Bump the version number using:

      .. code:: shell

         pixi run bump

--------------------------------------------------------------------------------

.. [1] This applies only if the Pixi environment is not activated already, *e.g.*
       by a ``direnv`` script.
