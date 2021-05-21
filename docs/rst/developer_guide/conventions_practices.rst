.. _sec-developer_guide-conventions_practices:

Conventions and development practices
=====================================

This pages briefly explains a few conventions and practices in the Eradiate
development team.

Style
-----

The Eradiate codebase is written following Python's
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_. Its code formatter of
choice is `Black <https://https://black.readthedocs.io/>`_ and its import
formatter of choice is `isort <https://pycqa.github.io/isort/>`_ (version 5 or
later), for which configuration files are provided at the root of the project.
Editor integration instructions are available
`for Black <https://black.readthedocs.io/en/stable/integrations/editors.html>`_
and `for isort <https://github.com/pycqa/isort/wiki/isort-Plugins>`_.

Code writing
------------

.. warning:: Eradiate is built using the `attrs <https://www.attrs.org>`_
   library. It is strongly recommended to read the ``attrs`` documentation prior
   to writing additional classes. In particular, it is important to understand
   the ``attrs`` initialisation sequence, as well as how callables can be used
   to set defaults and to create converters and validators.

   Eradiate's unit handling is based on `Pint <https://pint.readthedocs.io>`_,
   whose documentation is also a very helpful read.

   Finally, Eradiate uses custom Pint-based extensions to ``attrs`` now
   developed as the standalone project
   `Pinttrs <https://pinttrs.readthedocs.io>`_. Reading the Pinttrs docs is
   highly recommended.

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
    is usually better implemented as a class method constructor (typically
    ``from_something()``).

Initialisation from dictionaries
    Most of Eradiate's classes implement a special ``from_dict()`` class method
    constructor which initialises them from a dictionary. This method should
    interpret unit fields (see :func:`pinttr.interpret_units`).

    .. note:: The ``from_dict()`` class method is called by factories (see
       :ref:`sec-developer_guide-factory_guide`) based on the value of the
       dictionary's ``type`` field: it is therefore not advised to define a
       ``type`` field in new classes. Additional parameters are forwarded to the
       selected class's constructor as keyword arguments.

    By convention, we reserve the ``construct`` key for ``from_dict()`` to
    select a special constructor. Dictionary contents are then forwarded as
    keyword arguments to the designed class method.

    .. note:: Classes should therefore not have a ``construct`` field.

    .. admonition:: Example

       .. code:: python

          BiosphereFactory.create({
              # The following two parameters will select the
              # DiscreteCanopy.from_files() class method
              "type": "discrete_canopy",
              "construct": "from_files",
              # The following parameters will be passed as keyword arguments to
              # DiscreteCanopy.from_files()
              "id": "floating_spheres",
              "size": [100, 100, 30] * ureg.m,
              "leaf_cloud_dicts": [{
                  "instance_filename": instance_filename,
                  "leaf_cloud_filename": leaf_cloud_filename,
                  "leaf_reflectance": 0.4,
                  "leaf_transmittance": 0.1,
              }],
          })
