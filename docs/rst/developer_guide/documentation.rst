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

    cd $ERADIATE_DIR/docs
    make html

After the build is completed, the html document is located in
:code:`$ERADIATE_DIR/docs/_build/html`.

.. note:: By default, docs are built without running examples.
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

Our API is documenting using docstrings. We currently use a custom docstring
format which works well with our Sphinx theme, but we generally follow usual
practices; see documented content for examples.

Conventions
^^^^^^^^^^^

Docstrings start with a newline:

.. code:: python

   def my_func():
       """
       Docstring contents.
       """
       ...

Documenting attributes
^^^^^^^^^^^^^^^^^^^^^^

One notable exception to regular documentation practices is how we document
attributes. Eradiate ships a small documentation framework for that purpose
which notably allows to automatically create class docstrings for classes with
inherited attributes.

The :func:`.parse_docs` decorator must be applied to the documented class  prior
to any other action. Then, each declared attribute can be documented using the
:func:`.documented` function:

.. code:: python

   import attr
   from eradiate.util.attrs import parse_docs, documented

   @parse_docs  # Must be applied **after** attr.s
   @attr.s
   class MyClass:
       field = documented(
           attr.ib(default=None),
           doc="A documented attribute",
           type="float or None",
           default="None",
       )

The ``doc``, ``type`` and ``default`` parameters currently only support string
values.

Fields are sometimes partially redefined, but parts of their documentation can
be reused. For such cases, we provide the :func:`.get_doc` function:

.. code:: python

   from eradiate.util.attrs import get_doc

   @parse_docs
   @attr.s
   class MyChildClass(MyClass):
       field = documented(
           attr.ib(default=1.0),
           doc=get_doc(MyClass, "field", "doc"),
           type=get_doc(MyClass, "field", "type"),
           default="1.0",
       )

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
