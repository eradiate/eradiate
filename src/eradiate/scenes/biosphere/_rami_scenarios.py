from __future__ import annotations

import itertools
import typing
import zipfile
from enum import Enum
from pathlib import Path

from ._canopy_loader import load_scenario

DEFAULT_SCENARIO_FOLDER_NAME = ".scenarios"


class RAMIActualCanopies(Enum):
    JARVSELJA_PINE_STAND = "HET07_JPS_SUM"  # Summer
    OFENPASS_PINE_STAND = "HET08_OPS_WIN"  # Winter
    JARVSELJA_BIRCH_STAND_SUMMER = "HET09_JBS_SUM"  # Summer
    WELLINGTON_CITRUS_ORCHARD = "HET14_WCO_UND"
    JARVSELJA_BIRCH_STAND_WINTER = "HET15_JBS_WIN"  # Winter
    AGRICULTURAL_CROPS = "HET16_SRF_UND"  # Short Rotation Forest
    SAVANNA_PRE_FIRE = "HET50_SAV_PRE"  # Semi-empirical
    WYTHAM_WOOD = "HET51_WWO_TLS"  # Empirical


class RAMIHeterogeneousAbstractCanopies(Enum):
    ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_A = "HET10_DIS_S1A"
    ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_B = "HET11_DIS_S1B"
    ANISOTROPIC_BACKGROUND_OVERSTOREY_SPARSE_BRF_MODEL_C = "HET12_DIS_S1C"
    ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_A = "HET20_DIS_D1A"
    ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_B = "HET21_DIS_D1B"
    ANISOTROPIC_BACKGROUND_OVERSTOREY_DENSE_BRF_MODEL_C = "HET22_DIS_D1C"
    TWO_LAYER_CANOPY_OVERSTORIES_SPARSE_UNDERSTORIES_SPARSE = "HET16_DIS_S2S"
    TWO_LAYER_CANOPY_OVERSTORIES_MEDIUM_UNDERSTORIES_SPARSE = "HET17_DIS_M2S"
    TWO_LAYER_CANOPY_OVERSTORIES_DENSE_UNDERSTORIES_SPARSE = "HET18_DIS_D2S"
    TWO_LAYER_CANOPY_OVERSTORIES_SPARSE_UNDERSTORIES_DENSE = "HET26_DIS_S2D"
    TWO_LAYER_CANOPY_OVERSTORIES_MEDIUM_UNDERSTORIES_DENSE = "HET27_DIS_M2D"
    TWO_LAYER_CANOPY_OVERSTORIES_DENSE_UNDERSTORIES_DENSE = "HET28_DIS_D2D"
    CONSTANT_SLOPE_DISTRIBUTION_SPARSE_INCLINATION_15 = "HET23_DIS_S15"
    CONSTANT_SLOPE_DISTRIBUTION_DENSE_INCLINATION_15 = "HET24_DIS_D15"
    CONSTANT_SLOPE_DISTRIBUTION_SPARSE_INCLINATION_30 = "HET33_DIS_S30"
    CONSTANT_SLOPE_DISTRIBUTION_DENSE_INCLINATION_30 = "HET34_DIS_D30"


class RAMIHomogeneousAbstractCanopies(Enum):
    ANISOTROPIC_BACKGROUND_PLANOPHILE_A = "HOM23_DIS_P1A"
    ANISOTROPIC_BACKGROUND_PLANOPHILE_B = "HOM24_DIS_P1B"
    ANISOTROPIC_BACKGROUND_PLANOPHILE_C = "HOM25_DIS_P1C"
    # ANISOTROPIC_BACKGROUND_ERECTOPHILE_A = "HOM33_DIS_E1A"
    ANISOTROPIC_BACKGROUND_ERECTOPHILE_B = "HOM34_DIS_E1B"
    ANISOTROPIC_BACKGROUND_ERECTOPHILE_C = "HOM35_DIS_E1C"
    TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_DENSE = "HOM26_DIS_EPD"
    TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_MEDIUM = "HOM27_DIS_EPM"
    TWO_LAYER_CANOPY_ERECTOPHILE_SPARSE_PLANOPHILE_SPARSE = "HOM28_DIS_EPS"
    TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_DENSE = "HOM36_DIS_PED"
    TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_MEDIUM = "HOM37_DIS_PEM"
    TWO_LAYER_CANOPY_PLANOPHILE_SPARSE_ERECTOPHILE_SPARSE = "HOM38_DIS_PES"
    ADJACENT_CANOPIES_SPARSE_ERECTOPHILE_DENSE_PLANOPHILE = "HOM29_DIS_EM0"
    ADJACENT_CANOPIES_MEDIUM_ERECTOPHILE_SPARSE_PLANOPHILE = "HOM30_DIS_ED0"


class RAMIScenarioVersion(Enum):
    ORIGINAL = "original"
    SIMPLIFIED = "simplified"


def generate_name(
    scenario_name: (
        RAMIActualCanopies
        | RAMIHeterogeneousAbstractCanopies
        | RAMIHomogeneousAbstractCanopies
    ),
    version: RAMIScenarioVersion = RAMIScenarioVersion.ORIGINAL,
) -> str:
    """
    Generate a name for a scenario based on its name and version.

    Parameters
    ----------
    scenario_name : RAMIActualCanopies or RAMIHeterogeneousAbstractCanopies or RAMIHomogeneousAbstractCanopies
        The name of the scenario.
    version : ScenarioVersion
        The version of the scenario.

    Returns
    -------
    str
        The name of the scenario.
    """
    return (
        f"{scenario_name.value}-{version.value}"
        if version == RAMIScenarioVersion.SIMPLIFIED
        else scenario_name.value
    )


def _convert_to_enum(
    scenario_name: (
        str
        | RAMIActualCanopies
        | RAMIHeterogeneousAbstractCanopies
        | RAMIHomogeneousAbstractCanopies
    ),
) -> (
    RAMIActualCanopies
    | RAMIHeterogeneousAbstractCanopies
    | RAMIHomogeneousAbstractCanopies
):
    """
    Convert a scenario name to an enum if it is a string.

    Parameters
    ----------
    scenario_name : str or RAMIActualCanopies or RAMIHeterogeneousAbstractCanopies or RAMIHomogeneousAbstractCanopies
        The name of the scenario.

    Returns
    -------
    (RAMIActualCanopies | RAMIHeterogeneousAbstractCanopies | RAMIHomogeneousAbstractCanopies)
        The name of the scenario as an enum.
    """

    if isinstance(scenario_name, str):
        for member in itertools.chain.from_iterable(
            [
                RAMIActualCanopies,
                RAMIHeterogeneousAbstractCanopies,
                RAMIHomogeneousAbstractCanopies,
            ]
        ):
            if scenario_name == member.value:
                return member
        else:
            raise ValueError(f"Scenario {scenario_name} not found")
    else:
        return scenario_name


def load_rami_scenario(
    scenario_name: (
        str
        | RAMIActualCanopies
        | RAMIHeterogeneousAbstractCanopies
        | RAMIHomogeneousAbstractCanopies
    ),
    version: RAMIScenarioVersion = RAMIScenarioVersion.ORIGINAL,
    padding: int = 0,
    unpack_folder: typing.Optional[Path] = None,
) -> dict:
    """
    Load a scenario based on its name and version.

    Parameters
    ----------
    scenario_name : str or RAMIActualCanopies or RAMIHeterogeneousAbstractCanopies or RAMIHomogeneousAbstractCanopies
        The name of the RAMI-V scenario. If a string is provided, it will automatically be converted to the appropriate enum.
    version : RAMIScenarioVersion
        The version of the scenario.
    padding : int, optional
        The padding to apply to the scenario, defaults to 0.
    unpack_folder : Path, optional
        The folder to unpack the scenario to, defaults to None (current working directory).

    Returns
    -------
    dict
        The scenario.
    """
    from eradiate.data import data_store

    name = f"scenarios/rami5/{generate_name(_convert_to_enum(scenario_name), version)}"

    if unpack_folder is None:
        unpack_folder = Path.cwd() / DEFAULT_SCENARIO_FOLDER_NAME

    scenario_folder = unpack_folder / name
    if not scenario_folder.exists():
        local_path = data_store.fetch(f"{name}.zip")
        # Unzip the scenario
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(scenario_folder)

    # Load the scenario
    return load_scenario(scenario_folder, padding)
