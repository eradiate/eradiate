from pathlib import Path

import numpy as np
import pytest

import eradiate
from eradiate.experiments import CanopyExperiment
from eradiate.scenes.biosphere import load_scenario
from eradiate.scenes.biosphere._canopy_loader import apply_transformation
from eradiate.units import unit_registry as ureg

from eradiate.scenes.biosphere._rami_scenarios import (
    RAMIActualCanopies,
    RAMIHeterogeneousAbstractCanopies,
    RAMIHomogeneousAbstractCanopies,
    RAMIScenarioVersion,
    generate_name,
)

eradiate.set_mode("mono")


SAMPLE_SCENARIO_RPV = """{
  "surface": {
    "type": "rpv",
    "rho_0": {
        "wavelengths": [
          442.948,
          490.448,
          560.4304999999999,
          665.2445,
          681.556,
          709.1095,
          754.184,
          833.5,
          865.587,
          1242.5,
          1610.5,
          2120.0,
          2199.0
        ],
        "wavelengths_units": "nanometer",
        "values": [
          0.037938,
          0.04221,
          0.059786,
          0.088836,
          0.091155,
          0.096037,
          0.114102,
          0.120937,
          0.122402,
          0.147302,
          0.152672,
          0.152184,
          0.146692
        ],
        "type": "interpolated"
    },
    "g": {
        "wavelengths": [
          442.948,
          490.448,
          560.4304999999999,
          665.2445,
          681.556,
          709.1095,
          754.184,
          833.5,
          865.587,
          1242.5,
          1610.5,
          2120.0,
          2199.0
        ],
        "wavelengths_units": "nanometer",
        "values": [
          -0.25,
          -0.25,
          -0.25,
          -0.25,
          -0.25,
          -0.25,
          -0.2,
          -0.2,
          -0.2,
          -0.2,
          -0.2,
          -0.2,
          -0.2
        ],
        "type": "interpolated"
    },
    "k": {
        "wavelengths": [
          442.948,
          490.448,
          560.4304999999999,
          665.2445,
          681.556,
          709.1095,
          754.184,
          833.5,
          865.587,
          1242.5,
          1610.5,
          2120.0,
          2199.0
        ],
        "wavelengths_units": "nanometer",
        "values": [
          0.55,
          0.55,
          0.55,
          0.55,
          0.55,
          0.55,
          0.6,
          0.6,
          0.6,
          0.6,
          0.6,
          0.6,
          0.6
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
          "id": "Planophile",
          "mesh_tree_elements": [
            {
              "id": "Planophile",
              "mesh_filename": "tree-species/Planophile/Planophile.ply",
              "reflectance": {
                "quantity": "reflectance",
                "wavelengths": [
                  442.948,
                  490.448,
                  560.4304999999999,
                  665.2445,
                  681.556,
                  709.1095,
                  754.184,
                  833.5,
                  865.587,
                  1242.5,
                  1610.5,
                  2120.0,
                  2199.0
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
                  0.181612
                ],
                "type": "interpolated"
              },
              "transmittance": {
                "quantity": "transmittance",
                "wavelengths": [
                  442.948,
                  490.448,
                  560.4304999999999,
                  665.2445,
                  681.556,
                  709.1095,
                  754.184,
                  833.5,
                  865.587,
                  1242.5,
                  1610.5,
                  2120.0,
                  2199.0
                ],
                "wavelengths_units": "nanometer",
                "values": [
                  0.005325,
                  0.010844,
                  0.088037,
                  0.015605,
                  0.01291,
                  0.169364,
                  0.422238,
                  0.440022,
                  0.44141,
                  0.43188,
                  0.340786,
                  0.185841,
                  0.22086
                ],
                "type": "interpolated"
              }
            }
          ]
        },
        "instance_positions": [
          [
            [
              1.0,
              0.0,
              0.0,
              12.5
            ],
            [
              0.0,
              1.0,
              0.0,
              12.5
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
      25.0,
      25.0,
      1.0
    ]
  }
}
"""

SAMPLE_SCENARIO_SURFACE = """{
  "surface": {
    "type": "lambertian",
    "reflectance": {
      "quantity": "reflectance",
      "wavelengths": [
        442.95000000000005,
        490.45000000000005,
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
                "wavelengths": [
                  442.95000000000005,
                  490.45000000000005,
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
"""


def generate_expected_output_rpv(mesh_filename: Path):
    return {
        "canopy": {
            "instanced_canopy_elements": [
                {
                    "canopy_element": {
                        "id": "Planophile",
                        "mesh_tree_elements": [
                            {
                                "id": "Planophile",
                                "mesh_filename": mesh_filename,
                                "reflectance": {
                                    "quantity": "reflectance",
                                    "type": "interpolated",
                                    "values": pytest.approx(
                                        [
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
                                        ]
                                    ),
                                    "wavelengths": pytest.approx(
                                        [
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
                                        ]
                                    ),
                                    "wavelengths_units": "nanometer",
                                },
                                "transmittance": {
                                    "quantity": "transmittance",
                                    "type": "interpolated",
                                    "values": pytest.approx(
                                        [
                                            0.005325,
                                            0.010844,
                                            0.088037,
                                            0.015605,
                                            0.01291,
                                            0.169364,
                                            0.422238,
                                            0.440022,
                                            0.44141,
                                            0.43188,
                                            0.340786,
                                            0.185841,
                                            0.22086,
                                        ]
                                    ),
                                    "wavelengths": pytest.approx(
                                        [
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
                                        ]
                                    ),
                                    "wavelengths_units": "nanometer",
                                },
                            }
                        ],
                        "type": "mesh_tree",
                    },
                    "instance_positions": [pytest.approx([0.0, 0.0, 0.0])],
                }
            ],
            "padding": 0,
            "size": pytest.approx([25.0, 25.0, 1.0]),
            "type": "discrete_canopy",
        },
        "surface": {
            "g": {
                "type": "interpolated",
                "values": pytest.approx(
                    [
                        -0.25,
                        -0.25,
                        -0.25,
                        -0.25,
                        -0.25,
                        -0.25,
                        -0.2,
                        -0.2,
                        -0.2,
                        -0.2,
                        -0.2,
                        -0.2,
                        -0.2,
                    ]
                ),
                "wavelengths": pytest.approx(
                    [
                        442.948,
                        490.448,
                        560.4305,
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
                    ]
                ),
                "wavelengths_units": "nanometer",
            },
            "k": {
                "type": "interpolated",
                "values": pytest.approx(
                    [
                        0.55,
                        0.55,
                        0.55,
                        0.55,
                        0.55,
                        0.55,
                        0.6,
                        0.6,
                        0.6,
                        0.6,
                        0.6,
                        0.6,
                        0.6,
                    ]
                ),
                "wavelengths": pytest.approx(
                    [
                        442.948,
                        490.448,
                        560.4305,
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
                    ]
                ),
                "wavelengths_units": "nanometer",
            },
            "rho_0": {
                "type": "interpolated",
                "values": pytest.approx(
                    [
                        0.037938,
                        0.04221,
                        0.059786,
                        0.088836,
                        0.091155,
                        0.096037,
                        0.114102,
                        0.120937,
                        0.122402,
                        0.147302,
                        0.152672,
                        0.152184,
                        0.146692,
                    ]
                ),
                "wavelengths": pytest.approx(
                    [
                        442.948,
                        490.448,
                        560.4305,
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
                    ]
                ),
                "wavelengths_units": "nanometer",
            },
            "type": "rpv",
        },
    }


def generate_expected_output_surface(mesh_filename: Path):
    return {
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
                                "mesh_filename": mesh_filename,
                                "reflectance": {
                                    "quantity": "reflectance",
                                    "wavelengths": pytest.approx(
                                        [
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
                                            2199.0,
                                        ]
                                    ),
                                    "wavelengths_units": "nanometer",
                                    "values": pytest.approx(
                                        [
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
                                            0.394486,
                                        ],
                                    ),
                                    "type": "interpolated",
                                },
                            }
                        ],
                    },
                    "instance_positions": [pytest.approx([-33.0336, 45.8833, 0.0])],
                }
            ],
            "padding": 0,
            "size": pytest.approx([103.1214, 103.2308, 15.0213]),
        },
        "surface": {
            "type": "lambertian",
            "reflectance": {
                "quantity": "reflectance",
                "wavelengths": pytest.approx(
                    [
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
                        2199.0,
                    ]
                ),
                "wavelengths_units": "nanometer",
                "values": pytest.approx(
                    [
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
                        0.053261,
                    ]
                ),
                "type": "interpolated",
            },
        },
    }


class TestApplyTransformation:
    """Test case for the apply_transformation function."""

    @staticmethod
    def test_transformation():
        """Test the apply_transformation function with a specific transformation matrix and center point."""
        transf = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        center = np.array([1.0, 2.0, 0.0]) / 2
        expected_output = np.array([-0.5, -1.0, 0.0])
        assert np.allclose(apply_transformation(transf, center), expected_output)


class TestLoadScenario:
    """Test cases for the load_scenario function."""

    @staticmethod
    def test_load_scenario_rpv_without_spectral_data(tmp_path: Path):
        """Tests loading an RPV scenario without additional spectral data applied to the scenario elements."""
        scenario_folder = tmp_path

        (scenario_folder / "scenario.json").write_text(SAMPLE_SCENARIO_RPV)
        planophile_file = (
            scenario_folder / "tree-species" / "Planophile" / "Planophile.ply"
        )
        planophile_file.parent.mkdir(parents=True)
        planophile_file.touch()

        expected_output = generate_expected_output_rpv(planophile_file)
        parsed_scene = load_scenario(scenario_folder, padding=0)

        assert parsed_scene == expected_output

        # Check that the scene can be used to create an experiment
        CanopyExperiment(
            **parsed_scene,
            measures={
                "type": "perspective",
                "film_resolution": (10, 10),
                "origin": [1.0, 0.0, 0.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "sampler": "ldsampler",
                "fov": 20.0,
                "spp": 4,
                "srf": {
                    "type": "multi_delta",
                    "wavelengths": np.array([660, 550, 440]) * ureg.nm,
                },
            },
            illumination={"type": "directional", "zenith": 0.0, "azimuth": 0.0},
        )

    @staticmethod
    def test_load_scenario_surface_without_spectral_data(tmp_path: Path):
        """Tests loading a surface scenario without additional spectral data applied to the scenario elements."""
        scenario_folder = tmp_path

        (scenario_folder / "scenario.json").write_text(SAMPLE_SCENARIO_SURFACE)
        tree_file = (
            scenario_folder
            / "tree-species"
            / "Pinus-Montana"
            / "PIMO1"
            / "Pinus_mugo_PIMO1_-_Original"
            / "Wood.ply"
        )
        tree_file.parent.mkdir(parents=True)
        tree_file.touch()

        expected_output = generate_expected_output_surface(tree_file)

        parsed_scene = load_scenario(scenario_folder, padding=0)

        assert parsed_scene == expected_output
        # Check that the scene can be used to create an experiment
        CanopyExperiment(
            **parsed_scene,
            measures={
                "type": "perspective",
                "film_resolution": (10, 10),
                "origin": [1.0, 0.0, 0.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "sampler": "ldsampler",
                "fov": 20.0,
                "spp": 4,
                "srf": {
                    "type": "multi_delta",
                    "wavelengths": np.array([660, 550, 440]) * ureg.nm,
                },
            },
            illumination={"type": "directional", "zenith": 0.0, "azimuth": 0.0},
        )

    @staticmethod
    def test_load_scenario_rpv_with_spectral_data(tmp_path: Path):
        """Tests loading an RPV scenario with additional spectral data applied to the scenario elements."""
        scenario_folder = tmp_path

        (scenario_folder / "scenario.json").write_text(SAMPLE_SCENARIO_RPV)
        elem_file = scenario_folder / "tree-species" / "Planophile" / "Planophile.ply"
        elem_file.parent.mkdir(parents=True)
        elem_file.touch()

        expected_output = generate_expected_output_rpv(elem_file)

        spectral_data = {
            "Planophile": {
                "Planophile": {
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
        del expected_output["canopy"]["instanced_canopy_elements"][0]["canopy_element"][
            "mesh_tree_elements"
        ][0]["transmittance"]
        parsed_scene = load_scenario(
            scenario_folder, padding=0, spectral_data=spectral_data
        )

        assert parsed_scene == expected_output

        # Check that the scene can be used to create an experiment
        CanopyExperiment(
            **parsed_scene,
            measures={
                "type": "perspective",
                "film_resolution": (10, 10),
                "origin": [1.0, 0.0, 0.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "sampler": "ldsampler",
                "fov": 20.0,
                "spp": 4,
                "srf": {
                    "type": "multi_delta",
                    "wavelengths": np.array([660, 550, 440]) * ureg.nm,
                },
            },
            illumination={"type": "directional", "zenith": 0.0, "azimuth": 0.0},
        )

    @staticmethod
    def test_load_scenario_missing_scenario(tmp_path: Path):
        """Tests loading a scenario that does not exist."""
        scenario_folder = tmp_path
        padding = 0
        with pytest.raises(FileNotFoundError):
            load_scenario(scenario_folder, padding)


class TestGenerateName:
    @staticmethod
    def test_generate_name_actual_canopies_original():
        assert generate_name(RAMIActualCanopies.JARVSELJA_PINE_STAND) == "HET07_JPS_SUM"

    @staticmethod
    def test_generate_name_actual_canopies_simplified():
        assert (
            generate_name(
                RAMIActualCanopies.JARVSELJA_PINE_STAND, RAMIScenarioVersion.SIMPLIFIED
            )
            == "HET07_JPS_SUM-simplified"
        )

    @staticmethod
    def test_generate_name_heterogeneous_abstract_canopies_original():
        assert (
            generate_name(
                RAMIHeterogeneousAbstractCanopies.ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_A
            )
            == "HET10_DIS_S1A"
        )

    @staticmethod
    def test_generate_name_heterogeneous_abstract_canopies_simplified():
        assert (
            generate_name(
                RAMIHeterogeneousAbstractCanopies.ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_A,
                RAMIScenarioVersion.SIMPLIFIED,
            )
            == "HET10_DIS_S1A-simplified"
        )

    @staticmethod
    def test_generate_name_homogeneous_abstract_canopies_original():
        assert (
            generate_name(
                RAMIHomogeneousAbstractCanopies.ANISOTROPIC_BACKGROUND_PLANOPHILE_A
            )
            == "HOM23_DIS_P1A"
        )

    @staticmethod
    def test_generate_name_homogeneous_abstract_canopies_simplified():
        assert (
            generate_name(
                RAMIHomogeneousAbstractCanopies.ANISOTROPIC_BACKGROUND_PLANOPHILE_A,
                RAMIScenarioVersion.SIMPLIFIED,
            )
            == "HOM23_DIS_P1A-simplified"
        )
