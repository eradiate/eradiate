RAMI benchmark scenes
=====================

Eradiate ships a collection of canopy scenes built from the
`RAMI-V scene list <https://rami-benchmark.jrc.ec.europa.eu/_www/phase_descr.php?strPhase=RAMI5>`_
These pre-configured scenes are available for download request via the
datastore.\ [#sn2]_
Usage is documented in the
:ref:`scene loader guide <sec-user_guide-canopy_scene_loader-rami_scenes>`.

.. [#sn2] We thank the `DART <https://dart.omp.eu/>`_ team for letting us use
   their 3D model files to derive our scene models.

.. note::

   The renders in the index table below use material spectra interpolated from
   the monochromatic optical properties provided in the RAMI scene
   specifications. Colours are therefore not realistic.

.. list-table:: RAMI benchmark scene index
    :widths: 1 1 1 2
    :header-rows: 1

    * - RAMI ID
      - Description
      - Comments
      - Render
    {% for scene in scenes %}
    * - {{scene.rami_id}}
      - {{scene.description}}
      - {{scene.comments}}
      - .. image:: /fig/rami_scenes/{{scene.image_file}}
    {% endfor %}
