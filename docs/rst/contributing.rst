.. _sec-contributing:

Contributing to Eradiate
========================

..  TODO: Add "Where to start" section

.. important::
   Eradiate is written and documented in English using
   `Oxford spelling <https://en.wikipedia.org/wiki/Oxford_spelling>`_.


.. _sec-contributing-documentation:

Contributing to the documentation
---------------------------------

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once Eradiate is installed, the documentation can be built using the following
commands:

.. code-block:: bash

    cd $ERADIATE_SOURCE_DIR
    make docs

After the build is completed, the html document is located in
:code:`$ERADIATE_SOURCE_DIR/docs/_build/html`.

.. note::
   Some parts of the API documentation use static intermediate files generated
   by a dedicated script. See :ref:`sec-contributing-documentation-api-build`
   for more information.

Editing the API documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our API is documented using docstrings. We follow the
`Numpy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_,
with a few changes and updates documented hereafter.

Conventions
***********

Docstrings start with a newline:

.. code:: python

   def my_func():
       """
       Docstring contents.
       """
       ...

Documenting classes
*******************

In addition to the sections defined in the Numpy style guide, we add a "Fields"
section to our class docstrings. Class docstrings therefore have the following
structure:

Short summary
    A one-line summary that does not use variable names or the function name.
    It is notably printed in summary tables.
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#short-summary>`__.
Deprecation warning
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#deprecation-warning>`__.
Extended Summary
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#extended-summary>`__.
Parameters
    Description of the function arguments, keywords and their respective types.
    This section documents constructor parameters. Note that argument types
    should reflect types expected by the constructor, which can be broader
    than field types thanks to the ``attrs`` initialisation sequence.
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#parameters>`__.
Fields
    Description of class attributes. This section replaces the *Attributes*
    section defined in the Numpy style guide. Since Eradiate uses ``attrs``,
    fields are usually very similar to constructor parameters and rendered with
    the same style. Types indicated in this section should reflect the true
    field type, after applying converters. We use dedicated utility functions
    to generate the *Parameters* and *Fields* sections from in-source
    documentation (see below).

    Important *don'ts*:

    * Properties are documented automatically by the autosummary extension: do
      not document them in this section, they will be displayed in a dedicated
      *Attributes* rubric on the class documentation page.
    * Do not use *ivar* to document attributes: use this section instead.
    * Do not use the *Methods* section.

Returns
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#returns>`__.
Yields
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#yields>`__.
Receives
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#receives>`__.
Other Parameters
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#other-parameters>`__.
Raises
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#raises>`__.
Warns
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#warns>`__.
Warnings
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#warnings>`__.
See Also
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#see-also>`__.
Warns
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#warns>`__.
Notes
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#notes>`__.
References
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#references>`__.
Examples
    `See the Numpy docstring style guide for more detail <https://numpydoc.readthedocs.io/en/latest/format.html#examples>`__.

Field documentation helpers
***************************

Fields are documented using specific helper functions provided as part of
Eradiate' documentation framework. They notably allow to automatically create
class docstrings for classes with inherited fields.

The :func:`.parse_docs` decorator must be applied to the documented class  prior
to any other action. Then, each declared attribute can be documented using the
:func:`.documented` function:

.. code:: python

   import attr
   from typing import Optional
   from eradiate.util.attrs import parse_docs, documented

   @parse_docs  # Must be applied **after** attr.s
   @attr.s
   class MyClass:
       field: Optional[float] = documented(
           attr.ib(default=None),
           doc="A documented attribute",
           type="float, optional",
           default="None",
       )

In addition, a ``init_type`` argument lets the user specify if constructor
argument types are different from the field type. This is particularly useful
when a converter is systematically applied to field values upon initialisation:

.. code:: python

   import attr
   import numpy as np
   from eradiate.util.attrs import parse_docs, documented

   @parse_docs  # Must be applied **after** attr.s
   @attr.s
   class MyClass:
       field: np.ndarray = documented(
           attr.ib(converter=np.array),
           doc="A documented attribute",
           type="ndarray",
           init_type="array-like",
       )

The ``doc``, ``type``, ``init_type`` and ``default`` parameters currently only
support string values.

Fields are sometimes partially redefined, but parts of their documentation can
be reused. For such cases, we provide the :func:`.get_doc` function:

.. code:: python

   import attr
   from eradiate.util.attrs parse_docs, documented, import get_doc

   @parse_docs
   @attr.s
   class MyChildClass(MyClass):
       field = documented(
           attr.ib(default=1.0),
           doc=get_doc(MyClass, "field", "doc"),
           type=get_doc(MyClass, "field", "type"),
           default="1.0",
       )

.. _sec-contributing-documentation-api-build:

Building API RST files
**********************

Parts of the API documentation are generated using a dedicated Python script.
The generation process is integrated in the Sphinx configuration, but it can
sometimes be useful to build those static files manually. This can be done with
the ``docs-rst`` make target:

.. code-block:: bash

    cd $ERADIATE_SOURCE_DIR
    make docs-rst

Editing tutorials
^^^^^^^^^^^^^^^^^

Eradiate comes with tutorials shipped as Jupyter notebooks, saved to the
"`tutorials <https://github.com/eradiate/eradiate-tutorials>`_\ " submodule.
They are integrated in this documentation using the
`nbsphinx <https://nbsphinx.readthedocs.io/>`_ extension.

The recommended way to edit tutorials is as follows:

1. Open a terminal and start a Jupyter session.
2. In another terminal, open a Sphinx server using the following command at the
   root of your local copy of Eradiate:

   .. code:: bash

      make docs-serve

3. Browse to the tutorial you want to edit or create a new one using the
   tutorial template. You can now edit the content and see how it renders
   dynamically.

   .. important::

      Make sure that the first cell is as follows:

      .. code:: bash

         %reload_ext eradiate.notebook.tutorials

Nbsphinx renders markdown cells, but also allows to define raw reST cells, which
then support all usual Sphinx features (references, admonitions, etc.). See
`the documentation <https://nbsphinx.readthedocs.io/en/latest/raw-cells.html>`_
for more detail.

Tutorials are currently not run as part of the documentation build process;
instead, the output of the rendered notebook is checked in to the Git
repository. The reason for this is that rendering tutorials when building the
documentation would require a fully functional copy of Eradiate, including its
radiometric kernel Mitsuba. This is currently unachievable on the Read the Docs
service we use to deploy automatically the documentation upon committing to
GitHub: Mitsuba must be compiled and Read the Docs does not support its build
process.

.. important::

   Once you're done editing a tutorial, do not forget to rerun it entirely in a
   clean Jupyter session to render it as if you were a user.

Thumbnail galleries are not trivial difficult to fine-tune. The following pages
are useful when working on them:

* Galleries are created in Markdown/reST files following
  `these instructions <https://nbsphinx.readthedocs.io/en/latest/a-normal-rst-file.html#thumbnail-galleries>`_.
* Thumbnail selection is done following the
  recommendations of the nbsphinx documentation:
  `cell tag based <https://nbsphinx.readthedocs.io/en/latest/gallery/cell-tag.html>`_,
  `cell metadata based <https://nbsphinx.readthedocs.io/en/latest/gallery/cell-metadata.html>`_,
  `Sphinx config based <https://nbsphinx.readthedocs.io/en/latest/gallery/thumbnail-from-conf-py.html>`_.


.. _sec-contributing-codebase:

Contributing to the code base
-----------------------------

Style
^^^^^

* The Eradiate codebase is written following Python's
  `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_. Its code formatter of
  choice is `Black <https://black.readthedocs.io/>`_ and its linter of choice is
  `ruff <https://github.com/charliermarsh/ruff>`_, for which a configuration is
  provided as part of the ``pyproject.toml`` file.
  Editor integration instructions are available
  `for Black <https://black.readthedocs.io/en/stable/integrations/editors.html>`_
  and `for ruff <https://beta.ruff.rs/docs/editor-integrations/>`_.
  Both tools are part of our `pre-commit <https://pre-commit.com/>`_ hook set,
  which we strong recommend to install.

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
^^^^^^^^^^^^

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
    :ref:`sec-developer_guides-factory_guide`). This, in practice, reserves
    the ``"type"`` and ``"construct"`` parameters, meaning that
    factory-registered classes cannot have ``type`` or ``construct`` fields.

    For classes unregistered to any factory, our convention is to implement
    dictionary-based initialisation as a ``from_dict()`` class method
    constructor. It should implement behaviour similar to what
    :meth:`.Factory.convert` does, *i.e.*:

    * interpret units using :func:`pinttr.interpret_units`;
    * [optional] if relevant, allow for class method constructor selection using
      the ``"construct"`` parameter.


.. _sec-contributing-codebase-deprecations_removals:

Deprecations and removals
^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate tries to remain backward-compatible when possible. Sometimes however,
compatibility must be broken. Following the recommended practice in the Python
community, removals are, whenever possible, preceded by a deprecation period
during which a deprecated component is still available, marked as such in the
documentation, and using it triggers a :class:`DeprecationWarning`.

This workflow is facilitated by components defined in the
:mod:`util.deprecation <eradiate.util.deprecation>` module, and in particular
the :func:`.deprecated` decorator. Be sure to use them when relevant.

.. _sec-contributing-codebase-testing:

Testing
^^^^^^^

Eradiate is shipped with a series of tests written with
`pytest <https://docs.pytest.org/en/latest/>`_.

At the highest level, there is a separation of tests for Mitsuba plugins which
are maintained in the Eradiate codebase and tests for Eradiate's high-level
code. The tests for Eradiate are then grouped by complexity. First unit tests
are executed, followed by system tests and finally regression tests.

Running the test suite
**********************

To run the test suite, invoke ``pytest`` with the following command:

.. code-block:: bash

    pytest tests

Testing guidelines
******************

Writing test specification
""""""""""""""""""""""""""

Eradiate's tests can be roughly categorised as follows:

* unit tests focus on the smallest testable units of code;
* system tests check the behaviour of entire applications;
* regression tests which compare simulation results with previous versions.

While categorising each individual test is not always an easy task, this
nomenclature highlights the fact that tests have varied degrees of complexity.
When the rationale, setup and course of action of a test is not obvious by
reading the corresponding source code, properly documenting it in a structured
way is crucial. For this reason, Eradiate defines a test description template to
be used for system and regression tests.

The test specification consists of three main parts:

1. the **description of the test rationale**;
2. the **details of the setup**, explaining, in prose, how a test is designed;
3. the **expected outcome** of the test, which describes based on what the test
   should pass or fail.

The following template can be copied to new test cases and the information
filled in as needed. Note that we strongly suggest using string literals
(prefixed with a ``r``) in order to avoid issues with escape sequences.

.. code-block:: python

    r"""
    Test title
    ==========

    :Description: This is the short description of the test case

    Rationale
    ---------

    This is some explanatory text

    * This section explains the details
    * Of how the test is implemented
    * It can contain math! :math:`e^{i\pi}=-1`

    Expected behaviour
    ------------------

    This section explains the expected result of the test and how it is asserted.

    * We assert that something was calculated
    * Additionally the result must be correct
    """

The test specification can hold any valid restructured text. A quick rundown on that can be found
`here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ .

Regression tests
^^^^^^^^^^^^^^^^

Eradiate's regression tests are designed to allow the monitoring of results over
time. Each test produces a NetCDF file with the current results as well as an
image containing plots and metrics, comparing the current version of Eradiate to
the reference results. The results of these tests can be archived for future
reference.

To run the regression tests isolated from the rest of the test suite, we
introduced the ``regression`` fixture. To run only the regression tests, invoke
pytest like this:

.. code-block:: bash

    pytest tests -m "regression" --artefact-dir <a directory of your choice>

The ``artefact_dir`` parameter defines the output directory in which the results
and plots will be placed. If the directory does not exist, it will be created.
The artefact directory defaults to ``./build/test_artefacts``, which is resolved
relative to the current working directory.

Adding new regression tests
***************************

Regression tests use a comparison framework providing interfaces for statistical
and other metric-based tests. Relevant components are listed in the API
reference [:mod:`eradiate.test_tools`].

These tests are based on comparing the results of a computation to a reference,
computed on a previous version of the code which was deemed correct by other
means.

To implement tests based on this framework, we provide helper classes which can
be imported from the :mod:`eradiate.test_tools.regression` module:

.. code-block:: python

    import eradiate.test_tools.regression as ttr

Within your test case, you then instantiate one of the subclasses:

.. code-block:: python

    result = your_eradiate_simulation()

    test = ttr.Chi2Test(
        value=result,
        reference="path/to/the/data-file/reference.nc",
        threshold=0.05,
        archive_filename="/path/for/file/output.nc",
    )

After running a simulation on an Eradiate scene, you provide the resulting dataset as well as a path to
the reference result to the helper class. Adding a threshold value, which may depend on the scenario and the chosen
metric, and a path and filename for the outputs generated by the class the test is ready.
To execute the test it exposes the :meth:`.RegressionTest.run` method, which handles computing the
metric, storing the results in the given path, and returns the test outcome as a boolean.

The test will store two NetCDF files and an image file with a visualisation of the
results in the directory given as ``archive_filename``. It will store the new result
and the reference in two files, adding *-result* and *-ref* suffixes to the provided
filename.

To handle the test result simply use an assertion:

.. code-block:: python

    assert test.run()

Analysing the results
*********************

If the test fails due to a significant difference between the reference and the result the output can help in analysis.
The reference data and the result are stored in two NetCDF files under the path given in ``archive_filename``, which can
be imported and used in python scripts for detailed analysis. Furthermore the test adds an overview plot made up of four
parts: A direct visualisation of the result and reference data on the same axis, the absolute and relative difference between
result and reference in their own axes and the numerical value of the chosen metric.

In case this difference stems from a change made to Eradiate, which significantly alters the code's behaviour, the
reference needs to be updated. In this case, replace the existing reference file in the data repository and create a
pull request for the maintainers to review and add.

In case the test fails due to a missing or non found reference, for example when adding a new test case, the helper
will not attempt to compute the metric at all. Instead it will output the simulation result as NetCDF under the given
path with the *-ref* suffix alongside a simple visualisation of the result. The output can then be added to the data
repository as mentioned above.

Test report
^^^^^^^^^^^

Optionally, test results may be visualised using a report generated with a tool
located on a
`dedicated repository <https://github.com/eradiate/eradiate-test-report>`_.

The report summarises test outcomes and generates detailed entries for tests
specified with the docstring format specified above.

The test specification of unit tests is not parsed for the test report and does
not have to comply with these guidelines. For those, a short explanation is
sufficient, but the three general parts mentioned above should still serve as a
guideline for relevant and helpful test specification.

.. _sec-contributing-tips:

Tips
====

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

Profiling
---------

Tests are a very opportunity to profile Eradiate. We recommend running tests
with `pytest-profiling <https://pypi.org/project/pytest-profiling/>`_ (see
documentation for usage instructions, it's basically about installing the
package then running pytest with the ``--profile`` option).

Profiling stats can then be visualised with
`SnakeViz <https://jiffyclub.github.io/snakeviz/>`_.
