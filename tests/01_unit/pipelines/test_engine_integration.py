"""Integration tests for pipeline engine.

These tests simulate realistic postprocessing workflows to ensure the
pipeline engine works correctly in practice.
"""

import numpy as np
import pytest
import xarray as xr

from eradiate.pipelines import Pipeline
from eradiate.pipelines import validation as pval


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
            post_funcs=[
                pval.validate_type(xr.DataArray),
                pval.validate_dataarray_dims(["x", "y"]),
            ],
        )

        # Normalize data
        pipeline.add_node(
            "normalized",
            func=lambda raw_data: (raw_data - raw_data.mean()) / raw_data.std(),
            dependencies=["raw_data"],
            description="Normalize data",
            post_funcs=[pval.validate_all_finite()],
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

    def test_workflow_with_bypassing(self):
        """Test workflow with bypassed intermediate data."""
        pipeline = Pipeline()

        pipeline.add_node("raw", lambda: np.array([1, 2, 3]))
        pipeline.add_node(
            "processed",
            lambda raw: raw * 2,
            dependencies=["raw"],
        )
        pipeline.add_node(
            "final",
            lambda processed: processed + 1,
            dependencies=["processed"],
        )

        # Bypass intermediate step
        results = pipeline.execute(
            outputs=["final"],
            inputs={"processed": np.array([10, 20, 30])},
        )

        assert np.array_equal(results["final"], np.array([11, 21, 31]))


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
        assert np.array_equal(simple_results["result"], np.array([2, 4, 6, 8, 10]))

        # Test complex mode
        complex_pipeline = build_pipeline("complex")
        complex_results = complex_pipeline.execute(outputs=["result"])
        assert np.array_equal(complex_results["result"], np.array([12, 14, 16, 18, 20]))


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
        assert results["mean"] == pytest.approx(np.mean(data))
        assert results["std"] == pytest.approx(np.std(data))


class TestSubgraphExtraction:
    """Test extracting and executing subgraphs."""

    def test_extract_specific_branch(self):
        """Test extracting specific branch of computation."""
        # Build full pipeline
        pipeline = Pipeline()

        pipeline.add_node("data", lambda: np.array([1, 2, 3]))

        # Branch 1
        pipeline.add_node("double", lambda data: data * 2, dependencies=["data"])
        pipeline.add_node(
            "double_plus_one", lambda double: double + 1, dependencies=["double"]
        )

        # Branch 2
        pipeline.add_node("triple", lambda data: data * 3, dependencies=["data"])
        pipeline.add_node(
            "triple_plus_one", lambda triple: triple + 1, dependencies=["triple"]
        )

        # Extract only the "double" branch
        subgraph = pipeline.extract_subgraph(["double_plus_one"])

        # Should only contain relevant nodes
        assert set(subgraph._nodes.keys()) == {"data", "double", "double_plus_one"}

        # Execute subgraph
        results = subgraph.execute(outputs=["double_plus_one"])
        assert np.array_equal(results["double_plus_one"], np.array([3, 5, 7]))


class TestValidationIntegration:
    """Test validation in realistic scenarios."""

    def test_validation_catches_errors(self):
        """Test that validation catches errors in pipeline."""
        pipeline = Pipeline()

        # Node that produces negative values
        pipeline.add_node("data", lambda: np.array([-1, 0, 1]))

        # Node with validation that should fail
        pipeline.add_node(
            "positive_data",
            lambda data: data,
            dependencies=["data"],
            post_funcs=[pval.validate_positive()],
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="must be positive"):
            pipeline.execute(outputs=["positive_data"])

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled when needed."""
        pipeline = Pipeline()

        # Node that produces negative values
        pipeline.add_node("data", lambda: np.array([-1, 0, 1]))

        # Node with validation
        pipeline.add_node(
            "positive_data",
            lambda data: data,
            dependencies=["data"],
            post_funcs=[pval.validate_positive()],
        )

        # Disable validation and execute
        pipeline.set_global_validation(False)
        results = pipeline.execute(outputs=["positive_data"])

        # Should succeed even with negative values
        assert np.array_equal(results["positive_data"], np.array([-1, 0, 1]))

    def test_selective_validation(self):
        """Test disabling validation for specific nodes."""
        pipeline = Pipeline()

        pipeline.add_node("data", lambda: np.array([-1, 0, 1]))

        # Node with validation disabled
        pipeline.add_node(
            "unchecked",
            lambda data: data,
            dependencies=["data"],
            post_funcs=[pval.validate_positive()],
            validate_enabled=False,
        )

        # Should succeed because validation is disabled for this node
        results = pipeline.execute(outputs=["unchecked"])
        assert np.array_equal(results["unchecked"], np.array([-1, 0, 1]))


class TestIntermediateOutputsIntegration:
    """Test inspecting intermediate values via outputs."""

    def test_debug_intermediate_values(self):
        """Test using intermediate outputs for debugging."""
        pipeline = Pipeline()

        # Build pipeline
        pipeline.add_node("data", lambda: np.array([1, 2, 3]))
        pipeline.add_node("step1", lambda data: data * 2, dependencies=["data"])
        pipeline.add_node("step2", lambda step1: step1 + 1, dependencies=["step1"])
        pipeline.add_node("step3", lambda step2: step2 * 3, dependencies=["step2"])

        # Request all intermediate values as outputs
        results = pipeline.execute(outputs=["step3", "data", "step1", "step2"])

        # Check intermediate values
        assert np.array_equal(results["data"], np.array([1, 2, 3]))
        assert np.array_equal(results["step1"], np.array([2, 4, 6]))
        assert np.array_equal(results["step2"], np.array([3, 5, 7]))
        assert np.array_equal(results["step3"], np.array([9, 15, 21]))

    def test_post_func_for_inspection(self):
        """Test using post_funcs for side-effect inspection."""
        captured = {}

        pipeline = Pipeline()
        pipeline.add_node("data", lambda: np.array([1, 2, 3]))
        pipeline.add_node(
            "step1",
            lambda data: data * 2,
            dependencies=["data"],
            post_funcs=[lambda x: captured.update({"step1": x.copy()})],
        )
        pipeline.add_node("step2", lambda step1: step1 + 1, dependencies=["step1"])

        pipeline.execute(outputs=["step2"])

        assert np.array_equal(captured["step1"], np.array([2, 4, 6]))


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
            post_funcs=[
                pval.validate_type(xr.DataArray),
                pval.validate_dataarray_dims(["w", "g", "y", "x"]),
            ],
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
            post_funcs=[
                pval.validate_dataarray_dims(["w", "y", "x"]),
                pval.validate_all_finite(),
            ],
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
