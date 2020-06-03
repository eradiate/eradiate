.. _sec-developer_guide-tutorials:

Adding a Jupyter notebook tutorial
==================================

Some Eradiate tutorials are written as Jupyter notebooks, for an easy way to include
code examples in the tutorial text. This chapter will introduce all steps required to
add a new notebook-based tutorial to Eradiate.

Writing and storing the notebook
--------------------------------

When the notebook is in a publishable state, it should be completely executed and
saved including all outputs.

.. note::

   We require the output to be saved alongside the notebook's code and text
   in order to avoid executing it when building the documentation. While this
   significantly improves docs build time, it requires to regenerate
   notebook output after an update.

Tutorial notebooks are stored in a subdirectory of the data submodule, located
at ``$ERADIATE_DIR/resources/data/notebooks/tutorials``. In this directory,
create a new subdirectory and copy there the executed notebook and all
additional files required to run it.

.. note::

   If the notebook was saved without its outputs, these can be added without
   opening it in the Jupyter Notebook interface. The Jupyter command
   line interface can be used from the terminal. The following command will
   execute a notebook's content and update it with the resulting output:

   .. code-block:: bash

      jupyter nbconvert --to notebook --execute --inplace my_notebook.ipynb

In addition, it is good practice to apply IPython session defaults to all
tutorial notebooks. To do so, you can create a symbolic link to the dedicated
IPython profile configuration file, located at
``$ERADIATE_DIR/resources/data/notebooks/ipython_config.py``. In the directory
where the notebook is stored, create a relative symbolic link as follows:

.. code-block:: bash

   ln -s ../../ipython_config.py


Adding the tutorial to the docs
-------------------------------

Including files outside of the Sphinx root directory requires a trick. In the
directory ``docs/tutorials/notebooks``, create a subdirectory to host
the tutorial contents. In there, create symbolic links to all the resources
required for the tutorial, *i.e.* the notebook itself and all its ancillary
files.

Afterwards the notebook can be added in the root doc ``index.rst``
in the section ``tutorials`` by referencing the path to the symbolic link.

.. note::

   Symlinking files individually is important. Symlinking the enclosing directory
   will likely raise errors during docs compilation if the notebook's output
   contains graphics.

Committing
----------

.. warning::

    Before attempting to commit a new tutorial notebook, ensure that the repository
    **including** all submodules is up-to-date. If in doubt, run the following git
    command:

    .. code-block:: bash

        git pull
        git submodule update --recursive

To finalise the process, both the notebook and the changed docs need to be committed.
First, the notebook must be committed to the ``data`` submodule. After that,
the changed docs can be committed to the Eradiate repository, including the changes to the
submodule.