"""
Utility functions for the Eradiate dependency management system.
"""

from copy import deepcopy
from typing import Literal


def resolve_package_manager(
    layered_yml: dict, manager: Literal["pip", "conda"]
) -> dict:
    """
    When relevant, filter out irrelevant dependency specifications in "packages"
    sections of the ``layered.yml`` file depending on the selected Python package
    manager.
    """
    result = {}

    for key, section in layered_yml.items():
        result[key] = deepcopy(
            section
        )  # We are going to mutate this, so we work on a copy

        package_list = result[key].get("packages", [])
        result[key]["packages"] = [
            package[manager] if isinstance(package, dict) else package
            for package in package_list
        ]

    return result
