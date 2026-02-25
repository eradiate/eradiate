``eradiate.pipelines``
======================

.. automodule:: eradiate.pipelines

Pipeline configuration
----------------------

Post-processing pipelines are configured using a dictionary specifying
parameters listed in the documentation of the
:func:`~eradiate.pipelines.definitions.build_pipeline` function, which
implements post-processing pipeline assembly.
Configuration generation is automated by the :func:`~eradiate.pipelines.config`
helper function.

Many nodes of the DAG defining the pipeline are tagged for filtering upon query.
Tags explicitly used in the pipeline setup and execution are listed in the table
below.

.. list-table:: Pipeline node tags
   :widths: 15 15 70
   :header-rows: 1

   * - Tag
     - Value
     - Description
   * - ``"final"``
     - ``"true"``
     - The node's output will be selected to be part of the default pipeline
       output.
   * - ``"kind"``
     -
     - Defines the type of variable the node's output corresponds to.
       Data with the ``"data"`` tag will be selected to be part of the
       default result dataset.
   * -
     - ``"data"``
     - The output is a data variable and can be assembled into a dataset.
   * -
     - ``"coord"``
     - The output is a coordinate variable.

.. autofunction:: eradiate.pipelines.config

Pipeline assembly (``eradiate.pipelines.definitions``)
------------------------------------------------------

.. automodule:: eradiate.pipelines.definitions
   :members:
   :autosummary:

Pipeline engine (``eradiate.pipelines.engine``)
-----------------------------------------------

.. automodule:: eradiate.pipelines.engine
   :members:
   :autosummary:

Pipeline logic (``eradiate.pipelines.logic``)
---------------------------------------------

.. automodule:: eradiate.pipelines.logic
   :members:
   :autosummary:
