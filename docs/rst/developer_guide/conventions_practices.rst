.. _sec-developer_guide-conventions_practices:

Conventions and development practices
=====================================

This pages briefly explains a few conventions and practices in the Eradiate
development team.

Style
-----

* The Eradiate codebase is written following Python's
  `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_. Its code formatter of
  choice is `Black <https://https://black.readthedocs.io/>`_ and its import
  formatter of choice is `isort <https://pycqa.github.io/isort/>`_ (version 5 or
  later), for which configuration files are provided at the root of the project.
  Editor integration instructions are available
  `for Black <https://black.readthedocs.io/en/stable/integrations/editors.html>`_
  and `for isort <https://github.com/pycqa/isort/wiki/isort-Plugins>`_.

* We write our docstrings following the
  `Numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  We use the ``"""``-on-separate-lines style:

  .. code:: python

     def func(x):
         """
         Do something.

         Further detail on what this function does.
         """

* We use type hints in our library code. We do not use type hints in test code
  in general.

Code writing
------------

.. warning::

   * Eradiate is built using the `attrs <https://www.attrs.org>`_
     library. It is strongly recommended to read the ``attrs`` documentation
     prior to writing additional classes. In particular, it is important to
     understand the ``attrs`` initialisation sequence, as well as how callables
     can be used to set defaults and to create converters and validators.
   * Eradiate's unit handling is based on `Pint <https://pint.readthedocs.io>`_,
     whose documentation is also a very helpful read.
   * Eradiate uses custom Pint-based extensions to ``attrs`` now developed as the
     standalone project `Pinttrs <https://pinttrs.readthedocs.io>`_. Reading the
     Pinttrs docs is highly recommended.
   * Eradiate uses factories based on the
     `Dessine-moi <https://dessinemoi.readthedocs.io>`_ library. Reading the
     Dessine-moi docs is recommended.

When writing code for Eradiate, the following conventions and practices should
be followed.

Prefer relative imports in library code
    We generally use relative imports in library code, and absolute imports in
    tests and application code.

Minimise class initialisation code
    Using ``attrs`` for class writing encourages to minimise the amount of
    complex logic implemented by constructors. Although ``attrs`` provides the
    ``__attrs_post_init__()`` method to do so, we try to avoid it as much as
    possible. If a constructor must perform special tasks, then this logic
    is usually better implemented as a *class method constructor* (*e.g.*
    ``from_something()``).

Initialisation from dictionaries
    A lot of Eradiate's classes can be instantiated using dictionaries. Most of
    them leverage factories for that purpose (see
    :ref:`sec-developer_guide-factory_guide` and
    :ref:`sec-developer_guide-scene_element_guide`). This, in practice, reserves
    the ``"type"`` and ``"construct"`` parameters, meaning that
    factory-registered classes cannot have ``type`` or ``construct`` fields.

    For classes unregistered to any factory, our convention is to implement
    dictionary-based initialisation as a ``from_dict()`` class method
    constructor. It should implement behaviour similar to what
    :meth:`.Factory.convert` does, *i.e.*:

    * interpret units using :func:`pinttr.interpret_units`;
    * [optional] if relevant, allow for class method constructor selection using
      the ``"construct"`` parameter.

Shallow submodule caveats
-------------------------

Eradiate uses Git submodules to ship some of its data. Over time, these can grow
and become large enough so that using a *shallow submodule*. Shallow clones
do not contain the entire history of the repository and are therefore more
lightweight, saving bandwidth upon cloning.

However, shallow clones can be difficult to work with, especially when one
starts branching. If a shallow submodule is missing a remote branch you'd expect
it to track,
`this post <https://stackoverflow.com/questions/23708231/git-shallow-clone-clone-depth-misses-remote-branches>`_
contains probably what you need to do:

.. code:: bash

   cd my-shallow-submodule
   git remote set-branches origin '*'
   git fetch -v
   git checkout the-branch-i-ve-been-looking-for
