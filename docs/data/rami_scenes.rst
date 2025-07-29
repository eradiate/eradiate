RAMI benchmark scenes
=====================

Eradiate ships a collection of canopy scenes built from the
`RAMI-V scene list <https://rami-benchmark.jrc.ec.europa.eu/_www/phase_descr.php?strPhase=RAMI5>`_
These pre-configured scenes are available for download on the
`Eradiate website <https://eradiate.eu/data/store/unstable/scenarios/rami5/>`_.\ [#sn2]_
Usage is documented in the
:ref:`scene loader guide <sec-user_guide-canopy_scene_loader-rami_scenes>`. Note
that the :func:`.load_rami_scenario` function can download scenario data
automatically.

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
    * - HOM23_DIS_P1A
      - anisotropic background planophile a
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_PLANOPHILE_A_30_90.png
    * - HOM24_DIS_P1B
      - anisotropic background planophile b
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_PLANOPHILE_B_30_90.png
    * - HOM25_DIS_P1C
      - anisotropic background planophile c
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_PLANOPHILE_C_30_90.png
    * - HOM34_DIS_E1B
      - anisotropic background erectophile b
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_ERECTOPHILE_B_30_90.png
    * - HOM35_DIS_E1C
      - anisotropic background erectophile c
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_ERECTOPHILE_C_30_90.png
    * - HOM26_DIS_EPD
      - two layer canopy erectophile sparse planophile dense
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_DENSE_30_90.png
    * - HOM27_DIS_EPM
      - two layer canopy erectophile sparse planophile medium
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_MEDIUM_30_90.png
    * - HOM28_DIS_EPS
      - two layer canopy erectophile sparse planophile sparse
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_SPARSE_30_90.png
    * - HOM36_DIS_PED
      - two layer canopy planophile sparse erectophile dense
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_DENSE_30_90.png
    * - HOM37_DIS_PEM
      - two layer canopy planophile sparse erectophile medium
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_MEDIUM_30_90.png
    * - HOM38_DIS_PES
      - two layer canopy planophile sparse erectophile sparse
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_SPARSE_30_90.png
    * - HOM29_DIS_EM0
      - adjacent canopies sparse erectophile dense planophile
      - 
      - .. image:: /_images/rami_scenes/ADJACENT_CANOPIES_SPARSE_ERECTOPHILE_DENSE_PLANOPHILE_30_90.png
    * - HOM30_DIS_ED0
      - adjacent canopies medium erectophile sparse planophile
      - 
      - .. image:: /_images/rami_scenes/ADJACENT_CANOPIES_MEDIUM_ERECTOPHILE_SPARSE_PLANOPHILE_30_90.png
    * - HET10_DIS_S1A
      - anisotropic background overstorey sparse brf model a
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_A_30_90.png
    * - HET11_DIS_S1B
      - anisotropic background overstorey sparse brf model b
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_B_30_90.png
    * - HET12_DIS_S1C
      - anisotropic background overstorey sparse brf model c
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_C_30_90.png
    * - HET20_DIS_D1A
      - anisotropic background overstorey dense brf model a
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_A_30_90.png
    * - HET21_DIS_D1B
      - anisotropic background overstorey dense brf model b
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_B_30_90.png
    * - HET22_DIS_D1C
      - anisotropic background overstorey dense brf model c
      - 
      - .. image:: /_images/rami_scenes/ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_C_30_90.png
    * - HET16_DIS_S2S
      - two layer canopy overstories sparse understories sparse
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_SPARSE_UNDERSTORIES_SPARSE_30_90.png
    * - HET17_DIS_M2S
      - two layer canopy overstories medium understories sparse
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_MEDIUM_UNDERSTORIES_SPARSE_30_90.png
    * - HET18_DIS_D2S
      - two layer canopy overstories dense understories sparse
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_DENSE_UNDERSTORIES_SPARSE_30_90.png
    * - HET26_DIS_S2D
      - two layer canopy overstories sparse understories dense
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_SPARSE_UNDERSTORIES_DENSE_30_90.png
    * - HET27_DIS_M2D
      - two layer canopy overstories medium understories dense
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_MEDIUM_UNDERSTORIES_DENSE_30_90.png
    * - HET28_DIS_D2D
      - two layer canopy overstories dense understories dense
      - 
      - .. image:: /_images/rami_scenes/TWO_LAYER_CANOPY_OVERSTORIES_DENSE_UNDERSTORIES_DENSE_30_90.png
    * - HET23_DIS_S15
      - constant slope distribution sparse inclination 15
      - 
      - .. image:: /_images/rami_scenes/CONSTANT_SLOPE_DISTRIBUTION_SPARSE_INCLINATION_15_30_90.png
    * - HET24_DIS_D15
      - constant slope distribution dense inclination 15
      - 
      - .. image:: /_images/rami_scenes/CONSTANT_SLOPE_DISTRIBUTION_DENSE_INCLINATION_15_30_90.png
    * - HET33_DIS_S30
      - constant slope distribution sparse inclination 30
      - 
      - .. image:: /_images/rami_scenes/CONSTANT_SLOPE_DISTRIBUTION_SPARSE_INCLINATION_30_30_90.png
    * - HET34_DIS_D30
      - constant slope distribution dense inclination 30
      - 
      - .. image:: /_images/rami_scenes/CONSTANT_SLOPE_DISTRIBUTION_DENSE_INCLINATION_30_30_90.png
    * - HET07_JPS_SUM
      - jarvselja pine stand
      - 
      - .. image:: /_images/rami_scenes/JARVSELJA_PINE_STAND_30_90.png
    * - HET08_OPS_WIN
      - ofenpass pine stand
      - 
      - .. image:: /_images/rami_scenes/OFENPASS_PINE_STAND_30_90.png
    * - HET09_JBS_SUM
      - jarvselja birch stand summer
      - 
      - .. image:: /_images/rami_scenes/JARVSELJA_BIRCH_STAND_SUMMER_30_90.png
    * - HET14_WCO_UND
      - wellington citrus orchard
      - 
      - .. image:: /_images/rami_scenes/WELLINGTON_CITRUS_ORCHARD_30_90.png
    * - HET15_JBS_WIN
      - jarvselja birch stand winter
      - 
      - .. image:: /_images/rami_scenes/JARVSELJA_BIRCH_STAND_WINTER_30_90.png
    * - HET16_SRF_UND
      - agricultural crops
      - 
      - .. image:: /_images/rami_scenes/AGRICULTURAL_CROPS_30_90.png
    * - HET50_SAV_PRE
      - savanna pre fire
      - 
      - .. image:: /_images/rami_scenes/SAVANNA_PRE_FIRE_30_90.png
    * - HET51_WWO_TLS
      - wytham wood
      - This version of the Wytham Wood scene uses data from the updated v2 dataset.
      - .. image:: /_images/rami_scenes/WYTHAM_WOOD_30_90.png
