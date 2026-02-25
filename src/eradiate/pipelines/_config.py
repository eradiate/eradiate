"""
Post-processing pipeline configuration utilities.
"""

from __future__ import annotations

import logging

import eradiate

from .._mode import Mode
from ..scenes.integrators import Integrator
from ..scenes.measure import Measure
from ..spectral import BandSRF

logger = logging.getLogger(__name__)


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
