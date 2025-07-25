``eradiate.pipelines``
======================

.. automodule:: eradiate.pipelines

Pipeline configuration
----------------------

Post-processing pipelines are configured using a dictionary specifying
parameters listed in the table below. This dictionary is used as the input of
the :func:`eradiate.pipelines.driver` function. It should be noted that
:func:`eradiate.pipelines.config` can be used to generate a configuration
dictionary easily.

.. list-table:: Pipeline configuration variables
   :widths: 20 10 70
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``add_viewing_angles``
     - :class:`bool`
     - Whether the measure provides viewing angles that can be mapped to film
       pixels.
   * - ``apply_spectral_response``
     - :class:`bool`
     - Whether we should apply SRF weighting (a.k.a convolution) to spectral
       variables.
   * - ``measure_distant``
     - :class:`bool`
     - Whether we are processing the results of a distant measure.
   * - ``mode_id``
     - :class:`str`
     - The ID of the Eradiate mode for which the pipeline is configured.
   * - ``var_name``, ``var_metadata``
     - :class:`str`
     - The name and metadata for the physical variable that is being processed.

Many nodes of the DAG defining the pipeline are tagged for filtering upon query.
Tags explicitly used in the pipeline setup and execution are listed in the table
below. The :func:`eradiate.pipelines.outputs` function leverages these tags to
generate a list of default output nodes when running the post-processing
pipeline.

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

Hamilton driver creation and manipulation
-----------------------------------------

.. note::
   The following entry points are defined in the ``eradiate.pipelines.core``
   module.

.. py:currentmodule:: eradiate.pipelines

.. autosummary::
   :toctree: generated/autosummary/

   config
   driver
   list_variables
   outputs

Pipeline logic (``eradiate.pipelines.logic``)
---------------------------------------------

.. automodule:: eradiate.pipelines.logic
   :members:
   :autosummary:
