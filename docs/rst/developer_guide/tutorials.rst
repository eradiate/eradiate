.. _sec-developer_guide-tutorials:

Editing tutorials
=================

Eradiates uses the `sphinx-gallery <https://sphinx-gallery.github.io/>`_
extension to provide runnable and commented tutorials. Tutorials are located
in the ``$ERADIATE_DIR/docs/examples/tutorials`` directory.

.. warning:: It is strongly recommended to read carefully the sphinx-gallery
   user guide before proceeding.

Conventions
-----------

* We use the ``# %%`` code splitter convention.
* Sub-gallery ordering is set in the ``conf.py``.
* Examples are sorting based on their filename: you'll have to rename all files
  to customise the ordering.
* Gallery and sub-gallery READMEs are written in ``.txt`` format.
* Gallery titles are level-1 titles (``=====``); sub-gallery titles are level-2
  titles (``-----``).

Shipping supplementary material
-------------------------------

Supplementary material (*e.g.* configuration files required to run examples) is
not directly handled by sphinx-gallery. If you want to provide a download link
to supplementary files, you can use Sphinx's |download role|_. If you do so, be
sure to provide paths relative to the source root directory (using a leading
``/``); otherwise, sphinx-gallery's processing will not allow to reference your
files correctly.

.. |download role| replace:: ``:download:`` role
.. _download role: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-download

Editing examples
----------------

We recommend using Visual Studio Code to edit your examples interactively, since
it allows for the interactive execution of code blocks in the style of a
Jupyter notebook.

.. tip:: Keep example headers minimal (just the title and possibly a brief
   summary sentence). Proper introductory content should already be written in
   a commented code block.

Referencing examples
--------------------

You can reference an example using its label, defined following
`sphinx-gallery's naming convention <https://sphinx-gallery.github.io/stable/advanced.html#know-your-gallery-files>`_.
For instance, an example located at
``$ERADIATE_DIR/docs/examples/tutorials/my_example.py`` will have the label
``sphx_glr_examples_generated_tutorials_my_example.py``.

.. warning:: Changing filenames will break references! Do not forget to
   rebuild the docs and fix references if you move or rename an example.

Compiling the examples
----------------------

Due to technical limitations of our automatic docs deployment workflow, we
currently disable example execution by default when building the documentation.
We however highly recommend building them when compiling the documentation
locally in order to make sure that they render correctly. To do so, you should
use the ``html-plot`` Make target:

.. code:: bash

   make html-plot

.. seealso:: `Rerunning stale examples <https://sphinx-gallery.github.io/stable/configuration.html#rerunning-stale-examples>`_