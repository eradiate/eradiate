.. _sec-developer_guide-documentation:

Documenting Eradiate
====================

Building the documentation
--------------------------

Eradiate's documentation consists of two separate documents:

- the current document;
- the Mitsuba kernel documentation.

Building this document
^^^^^^^^^^^^^^^^^^^^^^

Once Eradiate is installed, this document can be built using the following
commands:

.. code-block:: bash

    cd $ERADIATE_DIR
    make docs

After the build is completed, the html document is located in
:code:`$ERADIATE_DIR/docs/_build/html`.

.. admonition:: Notes
   :class: note

   * Some parts of the API documentation use static intermediate files generated
     by a dedicated script. See :ref:`sec-developer_guide-documentation-api-build`
     for more information.
   * By default, docs are built without running examples.
     See :ref:`sec-developer_guide-documentation-tutorials-running_examples` for
     more information.

Building the kernel documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate's Mitsuba kernel exposes a CMake target to build its documentation,
which can be accessed once CMake is set up, as described in
:ref:`sec-getting_started-install`.

Run the follwing commands:

.. code-block:: bash

    cd $ERADIATE_DIR/build
    ninja mkdoc

The compiled html documentation will then be located in :code:`$ERADIATE_DIR/build/html`.

Editing the API documentation
-----------------------------

Our API is documented using docstrings. We follow the
`Numpy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_,
with a few changes and updates documented hereafter.

Conventions
^^^^^^^^^^^

Docstrings start with a newline:

.. code:: python

   def my_func():
       """
       Docstring contents.
       """
       ...

Documenting classes
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _sec-developer_guide-documentation-api-build:

Building API RST files
^^^^^^^^^^^^^^^^^^^^^^

Parts of the API documentation are generated using a dedicated Python script.
While regenerating those pages is not always required, keeping them up-to-date
is recommended. They can be generated using the ``rst-api`` make target:

.. code-block:: bash

    cd $ERADIATE_DIR
    make docs-rst-api
    make docs

Editing tutorials
-----------------

Eradiates uses the `sphinx-gallery <https://sphinx-gallery.github.io/>`_
extension to provide runnable and commented tutorials. Tutorials are located
in the ``$ERADIATE_DIR/docs/examples/tutorials`` directory.

.. warning:: It is strongly recommended to read carefully the sphinx-gallery
   user guide before proceeding.

Conventions
^^^^^^^^^^^

* We use the ``# %%`` code splitter convention.
* Sub-gallery ordering is set in the ``conf.py``.
* Examples are sorting based on their filename: you'll have to rename all files
  to customise the ordering.
* Gallery and sub-gallery READMEs are written in ``.txt`` format.
* Gallery titles are level-1 titles (``=====``); sub-gallery titles are level-2
  titles (``-----``).

Shipping supplementary material
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supplementary material (*e.g.* configuration files required to run examples) is
not directly handled by sphinx-gallery. If you want to provide a download link
to supplementary files, you can use Sphinx's |download role|_. If you do so, be
sure to provide paths relative to the source root directory (using a leading
``/``); otherwise, sphinx-gallery's processing will not allow to reference your
files correctly.

.. |download role| replace:: ``:download:`` role
.. _download role: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-download

Editing examples
^^^^^^^^^^^^^^^^

We recommend using Visual Studio Code to edit your examples interactively, since
it allows for the interactive execution of code blocks in the style of a
Jupyter notebook.

.. tip:: Keep example headers minimal (just the title and possibly a brief
   summary sentence). Proper introductory content should already be written in
   a commented code block.

Referencing examples
^^^^^^^^^^^^^^^^^^^^

You can reference an example using its label, defined following
`sphinx-gallery's naming convention <https://sphinx-gallery.github.io/stable/advanced.html#know-your-gallery-files>`_.
For instance, an example located at
``$ERADIATE_DIR/docs/examples/tutorials/my_example.py`` will have the label
``sphx_glr_examples_generated_tutorials_my_example.py``.

.. warning:: Changing filenames will break references! Do not forget to
   rebuild the docs and fix references if you move or rename an example.

.. _sec-developer_guide-documentation-tutorials-running_examples:

Running the examples
^^^^^^^^^^^^^^^^^^^^

Due to technical limitations of our automatic docs deployment workflow, we
currently disable example execution by default when building the documentation.
We however highly recommend building them when compiling the documentation
locally in order to make sure that they render correctly. To do so, you should
use the ``html-plot`` Make target:

.. code:: bash

   make html-plot

.. seealso:: `Rerunning stale examples <https://sphinx-gallery.github.io/stable/configuration.html#rerunning-stale-examples>`_
