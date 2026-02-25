"""
Post-processing pipeline builder definitions.

This module contains components to imperatively assemble a
:class:`.Pipeline` from a configuration dictionary.
"""

from __future__ import annotations

from . import logic
from .engine import Pipeline
from .._mode import modes

_MODE_IDS_CKD = set(modes(lambda x: x.is_ckd))

_FINAL_DATA = {"final": True, "kind": "data"}
_FINAL_COORD = {"final": True, "kind": "coord"}


def build_pipeline(config: dict) -> Pipeline:
    """
    Build a post-processing pipeline from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary. Expected keys:

        ``mode_id`` : str
            Eradiate mode identifier.
        ``measure_distant`` : bool
            Whether the measure is a distant measure.
        ``add_viewing_angles`` : bool
            Whether to compute viewing angles.
        ``var_name`` : str
            Name of the processed physical variable.
        ``var_metadata`` : dict
            Metadata to attach to the variable data array.
        ``apply_spectral_response`` : bool
            Whether to apply SRF weighting.
        ``calculate_variance`` : bool
            Whether to compute variance.
        ``calculate_stokes`` : bool
            Whether to compute the full Stokes vector.

    Returns
    -------
    .Pipeline
        A configured pipeline ready for execution.
    """
    pipeline = Pipeline()
    mode_id = config["mode_id"]
    measure_distant = config["measure_distant"]
    add_viewing_angles = config["add_viewing_angles"]
    var_name = config["var_name"]
    apply_srf = config["apply_spectral_response"]
    calc_var = config["calculate_variance"]
    calc_stokes = config["calculate_stokes"]
    is_ckd = mode_id in _MODE_IDS_CKD

    # ------------------------------------------------------------------
    # viewing_angles node (optional)
    # When False, 'viewing_angles' is a virtual input set to None.
    # ------------------------------------------------------------------
    if add_viewing_angles:
        pipeline.add_node(
            "viewing_angles",
            func=lambda angles: logic.viewing_angles(angles),
            dependencies=["angles"],
            description="Compute viewing angles dataset",
            metadata=_FINAL_COORD,
        )

    # ------------------------------------------------------------------
    # spectral_response (optional)
    # ------------------------------------------------------------------
    if apply_srf:
        pipeline.add_node(
            "spectral_response",
            func=lambda srf: logic.spectral_response(srf),
            dependencies=["srf"],
            description="Evaluate spectral response function",
        )

    # ------------------------------------------------------------------
    # extract_irradiance — expands into irradiance + solar_angles
    # ------------------------------------------------------------------
    pipeline.add_node(
        "_extract_irradiance",
        func=lambda mode_id, illumination, spectral_grid: logic.extract_irradiance(
            mode_id, illumination, spectral_grid
        ),
        dependencies=["mode_id", "illumination", "spectral_grid"],
        description="Extract irradiance and solar angles",
        outputs={"irradiance": "irradiance", "solar_angles": "solar_angles"},
    )
    pipeline.get_node("irradiance").metadata.update(_FINAL_DATA)
    pipeline.get_node("solar_angles").metadata.update(_FINAL_COORD)

    # ------------------------------------------------------------------
    # gather_bitmaps — expands into spp, weights_raw, <var>_raw [+ m2_raw]
    # viewing_angles is either a real node (add_viewing_angles=True) or a
    # virtual input set to None via inputs dict.
    # ------------------------------------------------------------------
    gather_outputs = ["spp", "weights_raw", f"{var_name}_raw"]
    if calc_var:
        gather_outputs.append(f"{var_name}_m2_raw")

    def _gather_bitmaps_func(
        mode_id,
        var_name,
        var_metadata,
        calculate_variance,
        calculate_stokes,
        bitmaps,
        solar_angles,
        viewing_angles,
    ):
        return logic.gather_bitmaps(
            mode_id,
            var_name,
            var_metadata,
            calculate_variance,
            calculate_stokes,
            bitmaps,
            viewing_angles,
            solar_angles,
        )

    pipeline.add_node(
        "_gather_bitmaps",
        func=_gather_bitmaps_func,
        dependencies=[
            "mode_id",
            "var_name",
            "var_metadata",
            "calculate_variance",
            "calculate_stokes",
            "bitmaps",
            "solar_angles",
            "viewing_angles",
        ],
        description="Gather raw bitmaps into xarray arrays",
        outputs=gather_outputs,
    )

    # ------------------------------------------------------------------
    # moment2_to_variance → <var>_var_raw (optional)
    # ------------------------------------------------------------------
    if calc_var:
        _vn = var_name  # capture for closure

        def _m2_to_var(**kwargs):
            return logic.moment2_to_variance(
                kwargs[f"{_vn}_raw"],
                kwargs[f"{_vn}_m2_raw"],
                kwargs["spp"],
                kwargs["calculate_stokes"],
            )

        pipeline.add_node(
            f"{var_name}_var_raw",
            func=_m2_to_var,
            dependencies=[
                f"{var_name}_raw",
                f"{var_name}_m2_raw",
                "spp",
                "calculate_stokes",
            ],
            description="Compute variance from 2nd moment",
        )

    # ------------------------------------------------------------------
    # aggregate_ckd_quad → <var>  (always — no-op in mono)
    # ------------------------------------------------------------------
    _vn = var_name  # capture for closure

    def _aggregate_main(**kwargs):
        return logic.aggregate_ckd_quad(
            kwargs["mode_id"],
            kwargs[f"{_vn}_raw"],
            kwargs["spectral_grid"],
            kwargs["ckd_quads"],
            False,
        )

    pipeline.add_node(
        var_name,
        func=_aggregate_main,
        dependencies=["mode_id", f"{var_name}_raw", "spectral_grid", "ckd_quads"],
        description=f"Aggregate CKD quadrature → {var_name}",
        metadata=_FINAL_DATA,
    )

    if calc_var:

        def _aggregate_var(**kwargs):
            return logic.aggregate_ckd_quad(
                kwargs["mode_id"],
                kwargs[f"{_vn}_var_raw"],
                kwargs["spectral_grid"],
                kwargs["ckd_quads"],
                True,
            )

        pipeline.add_node(
            f"{var_name}_var",
            func=_aggregate_var,
            dependencies=[
                "mode_id",
                f"{var_name}_var_raw",
                "spectral_grid",
                "ckd_quads",
            ],
            description=f"Aggregate CKD quadrature → {var_name}_var",
            metadata=_FINAL_DATA,
        )

    # ------------------------------------------------------------------
    # radiosity  (sector_radiosity only)
    # Must be added before radiosity_srf which depends on it.
    # ------------------------------------------------------------------
    if var_name == "sector_radiosity":
        pipeline.add_node(
            "radiosity",
            func=lambda sector_radiosity: logic.radiosity(sector_radiosity),
            dependencies=["sector_radiosity"],
            description="Aggregate sector radiosity",
            metadata=_FINAL_DATA,
        )

    # ------------------------------------------------------------------
    # SRF nodes  (CKD + apply_srf only)
    # ------------------------------------------------------------------
    if is_ckd and apply_srf:

        def _make_srf_node(src_name):
            def _srf_func(**kwargs):
                return logic.apply_spectral_response(kwargs[src_name], kwargs["srf"])

            return _srf_func

        pipeline.add_node(
            f"{var_name}_srf",
            func=_make_srf_node(var_name),
            dependencies=[var_name, "srf"],
            description=f"Apply SRF → {var_name}_srf",
            metadata=_FINAL_DATA,
        )

        if var_name == "sector_radiosity":
            pipeline.add_node(
                "radiosity_srf",
                func=_make_srf_node("radiosity"),
                dependencies=["radiosity", "srf"],
                description="Apply SRF → radiosity_srf",
                metadata=_FINAL_DATA,
            )

        if measure_distant:
            pipeline.add_node(
                "irradiance_srf",
                func=_make_srf_node("irradiance"),
                dependencies=["irradiance", "srf"],
                description="Apply SRF → irradiance_srf",
                metadata=_FINAL_DATA,
            )

    # ------------------------------------------------------------------
    # albedo  (sector_radiosity + distant)
    # ------------------------------------------------------------------
    if var_name == "sector_radiosity" and measure_distant:
        pipeline.add_node(
            "albedo",
            func=lambda radiosity, irradiance: logic.compute_albedo(
                radiosity, irradiance
            ),
            dependencies=["radiosity", "irradiance"],
            description="Compute surface albedo",
            metadata=_FINAL_DATA,
        )

        if is_ckd and apply_srf:
            pipeline.add_node(
                "albedo_srf",
                func=lambda radiosity_srf, irradiance_srf: logic.compute_albedo(
                    radiosity_srf, irradiance_srf
                ),
                dependencies=["radiosity_srf", "irradiance_srf"],
                description="Compute surface albedo (SRF-weighted)",
                metadata=_FINAL_DATA,
            )

    # ------------------------------------------------------------------
    # bidirectional_reflectance → brdf + brf  (radiance + distant)
    # ------------------------------------------------------------------
    if var_name == "radiance" and measure_distant:
        pipeline.add_node(
            "_brdf_brf",
            func=lambda radiance, irradiance, calculate_stokes: (
                logic.compute_bidirectional_reflectance(
                    radiance, irradiance, calculate_stokes
                )
            ),
            dependencies=["radiance", "irradiance", "calculate_stokes"],
            description="Compute BRDF and BRF",
            outputs=["brdf", "brf"],
        )
        pipeline.get_node("brdf").metadata.update(_FINAL_DATA)
        pipeline.get_node("brf").metadata.update(_FINAL_DATA)

        if is_ckd and apply_srf:

            def _brdf_brf_srf(radiance_srf, irradiance_srf, calculate_stokes):
                result = logic.compute_bidirectional_reflectance(
                    radiance_srf, irradiance_srf, calculate_stokes
                )
                return {"brdf_srf": result["brdf"], "brf_srf": result["brf"]}

            pipeline.add_node(
                "_brdf_brf_srf",
                func=_brdf_brf_srf,
                dependencies=["radiance_srf", "irradiance_srf", "calculate_stokes"],
                description="Compute BRDF and BRF (SRF-weighted)",
                outputs=["brdf_srf", "brf_srf"],
            )
            pipeline.get_node("brdf_srf").metadata.update(_FINAL_DATA)
            pipeline.get_node("brf_srf").metadata.update(_FINAL_DATA)

    # ------------------------------------------------------------------
    # dlp  (Stokes only)
    # ------------------------------------------------------------------
    if calc_stokes:
        pipeline.add_node(
            "dlp",
            func=lambda radiance: logic.degree_of_linear_polarization(radiance),
            dependencies=["radiance"],
            description="Compute degree of linear polarization",
            metadata=_FINAL_DATA,
        )

        if apply_srf:
            pipeline.add_node(
                "dlp_srf",
                func=lambda radiance_srf: logic.degree_of_linear_polarization(
                    radiance_srf
                ),
                dependencies=["radiance_srf"],
                description="Compute DLP (SRF-weighted)",
                metadata=_FINAL_DATA,
            )

    return pipeline
