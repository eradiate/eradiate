"""
Integration tests for pipeline engine.

These tests simulate realistic postprocessing workflows to ensure the
pipeline engine works correctly in practice.
"""

import numpy as np
import xarray as xr

from eradiate.pipelines.engine import Pipeline


class TestSimplePostprocessing:
    """Test a simple postprocessing pipeline."""

    def test_basic_workflow(self):
        """Test basic data processing workflow."""
        pipeline = Pipeline()

        # Simulate raw data assembly
        def assemble_raw():
            return xr.DataArray(
                np.random.randn(3, 4),
                dims=["x", "y"],
                coords={"x": [0, 1, 2], "y": [0, 1, 2, 3]},
            )

        # Add raw data node
        pipeline.add_node(
            "raw_data",
            func=assemble_raw,
            description="Assemble raw data",
        )

        # Normalize data
        pipeline.add_node(
            "normalized",
            func=lambda raw_data: (raw_data - raw_data.mean()) / raw_data.std(),
            dependencies=["raw_data"],
            description="Normalize data",
        )

        # Extract subset
        pipeline.add_node(
            "subset",
            func=lambda normalized: normalized.isel(x=slice(0, 2)),
            dependencies=["normalized"],
            description="Extract subset",
            metadata={"final": "true"},
        )

        # Execute
        results = pipeline.execute(outputs=["subset"])
        assert "subset" in results
        assert results["subset"].dims == ("x", "y")
        assert len(results["subset"]["x"]) == 2


class TestConditionalPipeline:
    """Test conditional pipeline construction."""

    def test_mode_dependent_pipeline(self):
        """Test building different pipelines based on mode."""

        def build_pipeline(mode: str) -> Pipeline:
            pipeline = Pipeline()

            # Raw data
            pipeline.add_node("raw", lambda: np.array([1, 2, 3, 4, 5]))

            # Mode-dependent processing
            if mode == "simple":
                pipeline.add_node(
                    "result",
                    lambda raw: raw * 2,
                    dependencies=["raw"],
                    metadata={"final": "true"},
                )
            elif mode == "complex":
                pipeline.add_node(
                    "intermediate",
                    lambda raw: raw * 2,
                    dependencies=["raw"],
                )
                pipeline.add_node(
                    "result",
                    lambda intermediate: intermediate + 10,
                    dependencies=["intermediate"],
                    metadata={"final": "true"},
                )

            return pipeline

        # Test simple mode
        simple_pipeline = build_pipeline("simple")
        simple_results = simple_pipeline.execute(outputs=["result"])
        np.testing.assert_array_equal(simple_results["result"], [2, 4, 6, 8, 10])

        # Test complex mode
        complex_pipeline = build_pipeline("complex")
        complex_results = complex_pipeline.execute(outputs=["result"])
        np.testing.assert_array_equal(complex_results["result"], [12, 14, 16, 18, 20])


class TestMultiBranchPipeline:
    """Test pipeline with multiple branches."""

    def test_parallel_branches(self):
        """Test pipeline with parallel processing branches."""
        pipeline = Pipeline()

        # Root node
        pipeline.add_node("data", lambda: np.array([1, 2, 3, 4, 5]))

        # Branch 1: Statistics
        pipeline.add_node(
            "mean",
            lambda data: float(np.mean(data)),
            dependencies=["data"],
            description="Compute mean",
            metadata={"final": "true"},
        )

        pipeline.add_node(
            "std",
            lambda data: float(np.std(data)),
            dependencies=["data"],
            description="Compute standard deviation",
            metadata={"final": "true"},
        )

        # Branch 2: Transformations
        pipeline.add_node(
            "normalized",
            lambda data: (data - np.mean(data)) / np.std(data),
            dependencies=["data"],
            description="Normalize data",
            metadata={"final": "true"},
        )

        pipeline.add_node(
            "squared",
            lambda data: data**2,
            dependencies=["data"],
            description="Square data",
            metadata={"final": "true"},
        )

        # Execute all outputs
        results = pipeline.execute()

        assert "mean" in results
        assert "std" in results
        assert "normalized" in results
        assert "squared" in results

        # Check values
        data = np.array([1, 2, 3, 4, 5])
        np.testing.assert_allclose(results["mean"], np.mean(data))
        np.testing.assert_allclose(results["std"], np.std(data))


class TestComplexWorkflow:
    """Test a complex realistic workflow."""

    def test_full_postprocessing_workflow(self):
        """Test a complete postprocessing workflow."""
        # Simulate CKD mode processing
        pipeline = Pipeline()

        # Raw data from solver
        def gather_raw():
            # Simulate spectral data with g-points
            return xr.DataArray(
                np.random.randn(3, 2, 4, 5),  # (w, g, y, x)
                dims=["w", "g", "y", "x"],
                coords={
                    "w": [400, 500, 600],
                    "g": [0, 1],
                    "y": [0, 1, 2, 3],
                    "x": [0, 1, 2, 3, 4],
                },
            )

        pipeline.add_node(
            "raw",
            func=gather_raw,
            description="Gather raw data from solver",
        )

        # CKD quadrature aggregation
        def aggregate_ckd(raw):
            # Simple mean over g-points (real implementation is more complex)
            return raw.mean(dim="g")

        pipeline.add_node(
            "aggregated",
            func=aggregate_ckd,
            dependencies=["raw"],
            description="Aggregate CKD quadrature",
        )

        # Extract top-of-atmosphere
        pipeline.add_node(
            "toa",
            func=lambda aggregated: aggregated.isel(y=-1, drop=True),
            dependencies=["aggregated"],
            description="Extract TOA",
            metadata={"final": "true"},
        )

        # Compute mean over wavelength
        pipeline.add_node(
            "spectral_mean",
            func=lambda aggregated: aggregated.mean(dim="w"),
            dependencies=["aggregated"],
            description="Spectral mean",
            metadata={"final": "true"},
        )

        # Execute
        results = pipeline.execute(outputs=["toa", "spectral_mean"])

        # Verify results
        assert "toa" in results
        assert "spectral_mean" in results
        assert results["toa"].dims == ("w", "x")
        assert results["spectral_mean"].dims == ("y", "x")

    def test_workflow_with_metadata_tracking(self):
        """Test workflow with metadata for tracking final outputs."""
        pipeline = Pipeline()

        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node(
            "c", lambda a: a + 2, dependencies=["a"], metadata={"final": "true"}
        )
        pipeline.add_node(
            "d", lambda b: b + 3, dependencies=["b"], metadata={"final": "true"}
        )

        # Get nodes tagged as final
        final_nodes = [
            name
            for name, node in pipeline._nodes.items()
            if node.metadata.get("final") == "true"
        ]

        assert set(final_nodes) == {"c", "d"}

        # Execute only final outputs
        results = pipeline.execute(outputs=final_nodes)
        assert results == {"c": 3, "d": 5}
