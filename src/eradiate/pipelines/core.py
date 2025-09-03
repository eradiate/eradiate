"""
Post-processing pipeline access points.
"""

from __future__ import annotations

import importlib
import logging

from hamilton import telemetry
from hamilton.base import DictResult, SimplePythonGraphAdapter
from hamilton.driver import Driver
from rich.table import Table

import eradiate

from .._mode import Mode
from ..scenes.integrators import Integrator
from ..scenes.measure import Measure
from ..spectral import BandSRF

logger = logging.getLogger(__name__)

# Disable Hamilton telemetry (we don't want to bother our users with it)
telemetry.disable_telemetry()


def config(
    measure: Measure,
    mode: Mode | str | None = None,
    integrator: Integrator | None = None,
) -> dict:
    """
    Generate a pipeline configuration for a specific scene setup.

    Parameters
    ----------
    measure : .Measure

    mode : .Mode or str, optional
        Mode or mode ID for which the pipeline is configured. By default, the
        current active mode is used.

    integrator : .Integrator or None, optional
        Integrator used for the experiment; indicates whether the moment was
        calculated during the integration.

    Returns
    -------
    dict

    See Also
    --------
    :mod:`eradiate.pipelines`
    """
    result = {}

    # Which mode is selected?
    if mode is None:
        mode = eradiate.mode()
    if isinstance(mode, str):
        mode = eradiate.Mode.new(mode)
    result["mode_id"] = mode.id

    # Is the measure distant?
    result["measure_distant"] = measure.is_distant()

    # Does the measure provide viewing angle values?
    result["add_viewing_angles"] = hasattr(measure, "viewing_angles")

    # Which physical variable are we processing?
    result["var_name"], result["var_metadata"] = measure.var

    # Shall we apply spectral response function weighting (a.k.a convolution)?
    result["apply_spectral_response"] = isinstance(measure.srf, BandSRF)

    # Should we calculate the variance in the result?
    result["calculate_variance"] = (
        integrator.moment if integrator is not None else False
    )

    # Should we calculate the stokes vector?
    result["calculate_stokes"] = integrator.stokes if integrator is not None else False

    if result["calculate_stokes"] and result["var_name"] != "radiance":
        logger.warning("Calculating stokes components on measures other than radiance.")

    return result


def driver(config: dict, definition_module: str):
    """
    Create a Hamilton :class:`~hamilton.driver.Driver` instance using the
    specified configuration. The post-processing pipeline mutates defined on
    parameters.

    Parameters
    ----------
    config : dict
        A configuration dictionary specifying various parameters
        (see :mod:`~eradiate.pipelines` for details).

    definition_module : str
        The module path to the Hamilton definitions.

    Returns
    -------
    hamilton.Driver

    See Also
    --------
    :func:`.config`
    """
    # Force power user mode, required to mutate node names based on the
    # processed variable name
    config = {"hamilton.enable_power_user_mode": True, **config}

    # Instantiate the driver following the Generic result builder documentation
    # https://hamilton.dagworks.io/en/latest/reference/result-builders/Generic/
    module = importlib.import_module(definition_module)
    dict_builder = DictResult()
    adapter = SimplePythonGraphAdapter(dict_builder)
    drv = Driver(config, module, adapter=adapter)
    return drv


def list_variables(
    drv: Driver, as_table: bool = False
) -> list[tuple[str, str, str]] | Table:
    """
    List variables available in a pipeline. Configuration variables are omitted.

    Parameters
    ----------
    drv : hamilton.driver.Driver
         Pipeline driver to inspect.

    as_table : bool, default: False
        If ``True``, format the results as a table.

    Returns
    -------
    variables : list or rich.table.Table
        List of variables ordered by name. Each element contains the following
        entries:

        * name;
        * type;
        * whether the variable is an input parameter.
    """
    data = []

    for variable in drv.list_available_variables():
        if variable.name in drv.graph.config:
            continue

        line = (
            variable.name,
            variable.type.__name__,
            "x" if variable.is_external_input else "",
        )
        data.append(line)

    result = sorted(data, key=lambda x: (0 if x[2] == "x" else 1, x[0]))

    if not as_table:
        return result
    else:
        table = Table()
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Input")

        for line in result:
            table.add_row(*line)

        return table


def outputs(drv: Driver):
    """
    Generate a list of default outputs for a given pipeline. Only nodes
    with the ``"final": "true"`` and ``"kind": "data"`` tags are selected.
    """
    return [
        x.name
        for x in drv.list_available_variables()
        if x.tags.get("final") == "true" and x.tags.get("kind") == "data"
    ]
