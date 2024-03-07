.. _sec-user_guide-scene_loader:

Scene loader
============

This page presents the canopy scene loader. A scene is composed by a
``scenario.json`` file and a set of mesh files. The ``scenario.json`` file
contains the description of the scene, including the properties of the surface
and canopies within the scene. The mesh files contain the geometry of the canopy
elements.

The ``scenario.json`` file is structured to define specific properties of the
surface and canopies within a scene. It closely aligns with the expected scene
description format for the :class:`~eradiate.experiments.CanopyExperiment`
within its ``surface`` and ``canopy`` fields.

Below is an example ``scenario.json`` containing a Lambertian surface and a
discrete canopy composed of mesh tree elements:

.. code-block:: json-object

    {
      "surface": {
        "type": "lambertian",
        "reflectance": {
          "quantity": "reflectance",
          "wavelengths": [
            442.95,
            490.45,
            560.4,
            665.25,
            681.55,
            709.1,
            754.15,
            833.5,
            865.6,
            1242.5,
            1610.5,
            2120.0,
            2199.0
          ],
          "wavelengths_units": "nanometer",
          "values": [
            0.882742,
            0.897852,
            0.907004,
            0.910396,
            0.913883,
            0.902941,
            0.900788,
            0.856648,
            0.848049,
            0.368409,
            0.04536,
            0.016815,
            0.053261
          ],
          "type": "interpolated"
        }
      },
      "canopy": {
        "type": "discrete_canopy",
        "instanced_canopy_elements": [
          {
            "canopy_element": {
              "type": "mesh_tree",
              "id": "PIMO1",
              "mesh_tree_elements": [
                {
                  "id": "Wood",
                  "mesh_filename": "tree-species/Pinus-Montana/PIMO1/Pinus_mugo_PIMO1_-_Original/Wood.ply",
                  "reflectance": {
                    "quantity": "reflectance",
                    "wavelengths": {
                      "values": [
                        442.95,
                        490.45,
                        560.4,
                        665.25,
                        681.55,
                        709.1,
                        754.15,
                        833.5,
                        865.6,
                        1242.5,
                        1610.5,
                        2120.0,
                        2199.0
                      ],
                      "units": "nanometer",
                      "type": "interpolated"
                    },
                    "values": [
                      0.072764,
                      0.082621,
                      0.109325,
                      0.157819,
                      0.167218,
                      0.200941,
                      0.239725,
                      0.303587,
                      0.329257,
                      0.525503,
                      0.511729,
                      0.383173,
                      0.394486
                    ],
                    "type": "interpolated"
                  }
                }
              ]
            },
            "instance_positions": [
              [
                [
                  0.2813481616359169,
                  0.9596057586030265,
                  0.0,
                  18.5271
                ],
                [
                  -0.9596057586030265,
                  0.2813481616359169,
                  0.0,
                  97.4987
                ],
                [
                  0.0,
                  0.0,
                  1.0,
                  0.0
                ],
                [
                  0.0,
                  0.0,
                  0.0,
                  1.0
                ]
              ]
            ]
          }
        ],
        "size": [
          103.1214,
          103.2308,
          15.0213
        ]
      }
    }

The folder structure for the above example would be:

.. code-block:: bash

    .
    ├── scenario.json
    └── tree-species
        └── Pinus-Montana
            └── PIMO1
                └── Pinus_mugo_PIMO1_-_Original
                    └── Wood.ply

Differences from ``CanopyExperiment``
-------------------------------------

Mesh filename
    The ``mesh_filename`` field within the canopy element specifies the relative
    path to the mesh file. This path is relative to the ``scenario.json`` file
    itself. Upon loading the scene, this relative path will be expanded to the
    absolute path where the mesh file is located.

Instance positions
    Instance positions within the canopy configuration are described using a
    4x4 affine transformation matrix. This matrix details the transformations
    applied to position the original mesh within the scene.\ [#sn1]_

    .. [#sn1] The current implementation supports translations only. Support for rotations
       is foreseen in a future update.

Custom spectral properties
--------------------------

The ``reflectance`` field within the surface and canopy elements can be
customized using custom spectral properties. To do so, a dictionary describing
the spectral to be used is provided with matching ``canopy_element`` and
``mesh_tree_elements`` ids. An example of a custom spectral property is shown
below:

.. code-block:: python

   spectral_data = {
       "PIMO1": {
           "Wood": {
               "reflectance": (
                   {
                       "quantity": "reflectance",
                       "wavelengths": [
                           442.948,
                           490.448,
                           560.43045,
                           665.2445,
                           681.556,
                           709.1095,
                           754.184,
                           833.5,
                           865.587,
                           1242.5,
                           1610.5,
                           2120.0,
                           2199.0,
                       ],
                       "wavelengths_units": "nanometer",
                       "values": [
                           0.053892,
                           0.057882,
                           0.136485,
                           0.055265,
                           0.052734,
                           0.214271,
                           0.4771,
                           0.494542,
                           0.496112,
                           0.461875,
                           0.332809,
                           0.158912,
                           0.181612,
                       ],
                       "type": "interpolated",
                   }
               ),
           },
       },
   }


RAMI benchmark scenarios
------------------------

We provide a specific loader for scenes derived from the
`RAMI-V scene list <https://rami-benchmark.jrc.ec.europa.eu/_www/phase_descr.php?strPhase=RAMI5>`_.
These pre-configured are available for use and downloaded upon request via the
datastore.\ [#sn2]_ These are downloaded when a specific scenario is requested via the
datastore. Due to their size and the number of files they contain, the scenarios
are downloaded in a compressed format and, by default, extracted to the current
working directory. The extracted files are then used to load the scenario. To
change the default location for the extracted files, set the appropriate
parameter in the :func:`.load_rami_scenario` function.

.. [#sn2] We thank the `DART <https://dart.omp.eu/>`_ team for allowing us to
   use their 3D model files to derive our scene models.

.. code-block:: python

    from pathlib import Path

    import eradiate
    from eradiate.experiments import CanopyExperiment
    from eradiate.scenes.biosphere import load_rami_scenario
    from eradiate.units import unit_registry as ureg

    eradiate.set_mode("mono")

    scenario_data = load_rami_scenario("HOM30_DIS_ED0")

    scenario = CanopyExperiment(
        **scenario_data,
        measures={
            "type": "perspective",
            "film_resolution": (50, 50),
            "origin": [10.0, 10.0, 10.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "sampler": "ldsampler",
            "fov": 50.0,
            "spp": 4**2,
            "srf": {
                "type": "multi_delta",
                "wavelengths": np.array([660.0, 550.0, 440.0]) * ureg.nm,
            },
        },
        illumination={"type": "directional", "zenith": 45.0, "azimuth": 350.0},
    )

    res_eradiate = eradiate.run(scenario)
