"""Realistic postprocessing pipeline example.

This example shows a more realistic pipeline that mimics the kind of
postprocessing workflow used in Eradiate for radiative transfer data.
"""

import numpy as np
import xarray as xr

from eradiate.pipelines import Pipeline
from eradiate.pipelines import validation as pval


def build_postprocessing_pipeline(mode: str = "ckd", apply_srf: bool = False):
    """Build a realistic postprocessing pipeline.

    Parameters
    ----------
    mode : str
        Computation mode: "mono" or "ckd"
    apply_srf : bool
        Whether to apply spectral response function

    Returns
    -------
    Pipeline
        Configured pipeline
    """
    pipeline = Pipeline()

    # Step 1: Gather raw data from solver
    def gather_raw_data():
        """Simulate gathering raw data from DISORT solver."""
        # Simulate data with spectral (w), g-points (g), and spatial dimensions
        # Use non-negative values (radiance must be positive)
        if mode == "ckd":
            data = np.abs(np.random.randn(5, 4, 10, 8)) + 0.1  # (w, g, y, x)
            dims = ["w", "g", "y", "x"]
            coords = {
                "w": np.linspace(400, 800, 5),  # wavelength [nm]
                "g": np.arange(4),  # g-points
                "y": np.arange(10),  # viewing angles
                "x": np.arange(8),  # azimuth angles
            }
        else:  # mono
            data = np.abs(np.random.randn(5, 10, 8)) + 0.1  # (w, y, x)
            dims = ["w", "y", "x"]
            coords = {
                "w": np.linspace(400, 800, 5),
                "y": np.arange(10),
                "x": np.arange(8),
            }

        return xr.DataArray(data, dims=dims, coords=coords, name="radiance_raw")

    pipeline.add_node(
        "radiance_raw",
        func=gather_raw_data,
        description="Gather raw radiance from solver",
        post_funcs=[
            pval.validate_type(xr.DataArray),
            pval.validate_all_finite(),
        ],
    )

    # Step 2: CKD quadrature aggregation (only in CKD mode)
    if mode == "ckd":

        def aggregate_ckd_quad(radiance_raw):
            """Aggregate over CKD g-points using quadrature."""
            # Simple mean over g-points
            return radiance_raw.mean(dim="g")

        pipeline.add_node(
            "radiance",
            func=aggregate_ckd_quad,
            dependencies=["radiance_raw"],
            description="Aggregate CKD quadrature",
            post_funcs=[
                pval.validate_dataarray_dims(["w", "y", "x"]),
                pval.validate_all_finite(),
            ],
        )
    else:
        # In mono mode, just pass through
        pipeline.add_node(
            "radiance",
            func=lambda radiance_raw: radiance_raw,
            dependencies=["radiance_raw"],
            description="Pass through (mono mode)",
        )

    # Step 3: Apply spectral response function (optional)
    if apply_srf:

        def apply_spectral_response(radiance):
            """Apply spectral response function (convolution)."""
            # Simplified: just apply Gaussian weighting
            weights = np.exp(-0.5 * ((radiance.w.values - 550) / 50) ** 2)
            weights = weights / weights.sum()
            # Create DataArray with proper dimensions for broadcasting
            weights_da = xr.DataArray(weights, dims=["w"], coords={"w": radiance.w})
            return (radiance * weights_da).sum(dim="w")

        pipeline.add_node(
            "radiance_srf",
            func=apply_spectral_response,
            dependencies=["radiance"],
            description="Apply spectral response function",
            post_funcs=[pval.validate_all_finite()],
            metadata={"final": "true", "kind": "data"},
        )

    # Step 4: Compute BRDF (normalize by irradiance)
    def compute_irradiance():
        """Compute reference irradiance."""
        # Simplified constant irradiance
        return 1000.0  # W/m^2

    pipeline.add_node(
        "irradiance",
        func=compute_irradiance,
        description="Compute reference irradiance",
    )

    def compute_brdf(radiance, irradiance):
        """Compute bidirectional reflectance."""
        return radiance / (irradiance * np.pi)

    pipeline.add_node(
        "brdf",
        func=compute_brdf,
        dependencies=["radiance", "irradiance"],
        description="Compute BRDF",
        post_funcs=[
            pval.validate_non_negative(),
            pval.validate_all_finite(),
        ],
        metadata={"final": "true", "kind": "data"},
    )

    # Step 5: Compute spectral mean (for quick analysis)
    pipeline.add_node(
        "radiance_spectral_mean",
        func=lambda radiance: radiance.mean(dim="w"),
        dependencies=["radiance"],
        description="Compute spectral mean",
        metadata={"final": "true", "kind": "statistics"},
    )

    return pipeline


def main():
    """Run realistic pipeline example."""
    print("\n" + "=" * 70)
    print("Realistic Postprocessing Pipeline Example")
    print("=" * 70 + "\n")

    # Build CKD pipeline with SRF
    print("Building CKD pipeline with SRF...")
    pipeline_ckd_srf = build_postprocessing_pipeline(mode="ckd", apply_srf=True)

    # Print summary
    pipeline_ckd_srf.print_summary()

    # Export visualizations
    print("\nExporting visualizations...")

    # DOT format
    dot_file = "examples/realistic_pipeline.dot"
    pipeline_ckd_srf.write_dot(dot_file)
    print(f"✓ Exported DOT: {dot_file}")

    # Execute pipeline
    print("\nExecuting pipeline...")
    results = pipeline_ckd_srf.execute()

    print("\nPipeline executed successfully!")
    print("Outputs generated:")
    for name, data in results.items():
        if isinstance(data, xr.DataArray):
            print(f"  - {name}: {data.dims}, shape={data.shape}")
        else:
            print(f"  - {name}: {type(data).__name__} = {data}")

    # Show how to extract subgraph
    print("\n" + "-" * 70)
    print("Extracting subgraph for BRDF computation only...")
    brdf_pipeline = pipeline_ckd_srf.extract_subgraph(["brdf"])
    print(f"Subgraph nodes: {brdf_pipeline.list_nodes()}")

    # Show how to bypass expensive computation
    print("\n" + "-" * 70)
    print("Bypassing raw data gathering (for testing)...")
    mock_radiance_raw = xr.DataArray(
        np.abs(np.random.randn(5, 4, 10, 8)) + 0.1,  # Non-negative
        dims=["w", "g", "y", "x"],
        coords={
            "w": np.linspace(400, 800, 5),
            "g": np.arange(4),
            "y": np.arange(10),
            "x": np.arange(8),
        },
    )

    results_bypassed = pipeline_ckd_srf.execute(
        outputs=["brdf"], inputs={"radiance_raw": mock_radiance_raw}
    )
    print("Bypassed execution successful!")
    print(f"Output: brdf, shape={results_bypassed['brdf'].shape}")

    # Compare with mono mode
    print("\n" + "-" * 70)
    print("Building mono mode pipeline for comparison...")
    pipeline_mono = build_postprocessing_pipeline(mode="mono", apply_srf=False)
    print(f"Mono pipeline nodes: {len(pipeline_mono._nodes)}")
    print(f"CKD+SRF pipeline nodes: {len(pipeline_ckd_srf._nodes)}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nTo visualize the pipeline graph:")
    print(f"  dot -Tpng {dot_file} -o realistic_pipeline.png")
    print(f"  dot -Tsvg {dot_file} -o realistic_pipeline.svg")
    print("\n")


if __name__ == "__main__":
    main()
