"""Demonstration of the pipeline engine.

This script shows how to use the pipeline engine for a simple data processing
workflow with validation, interception, and bypassing.
"""

import numpy as np
import xarray as xr

from eradiate.pipelines import Pipeline
from eradiate.pipelines import validation as pval


def demo_basic_pipeline():
    """Demonstrate basic pipeline creation and execution."""
    print("=" * 60)
    print("Demo 1: Basic Pipeline")
    print("=" * 60)

    pipeline = Pipeline()

    # Add simple nodes
    pipeline.add_node("a", lambda: 10, description="Constant 10")
    pipeline.add_node("b", lambda: 20, description="Constant 20")
    pipeline.add_node(
        "sum",
        lambda a, b: a + b,
        dependencies=["a", "b"],
        description="Add a and b",
    )
    pipeline.add_node(
        "product",
        lambda a, b: a * b,
        dependencies=["a", "b"],
        description="Multiply a and b",
    )

    # Execute
    results = pipeline.execute(outputs=["sum", "product"])
    print(f"Results: {results}")
    print()


def demo_xarray_pipeline():
    """Demonstrate pipeline with xarray DataArrays and validation."""
    print("=" * 60)
    print("Demo 2: Pipeline with xarray and Validation")
    print("=" * 60)

    pipeline = Pipeline()

    # Create raw data
    def create_raw_data():
        return xr.DataArray(
            np.random.randn(3, 4, 5),
            dims=["x", "y", "z"],
            coords={
                "x": [0, 1, 2],
                "y": [0, 1, 2, 3],
                "z": [0, 1, 2, 3, 4],
            },
        )

    pipeline.add_node(
        "raw",
        create_raw_data,
        description="Create raw data",
        post_funcs=[
            pval.validate_type(xr.DataArray),
            pval.validate_dataarray_dims(["x", "y", "z"]),
            pval.validate_all_finite(),
        ],
    )

    # Normalize
    pipeline.add_node(
        "normalized",
        lambda raw: (raw - raw.mean()) / raw.std(),
        dependencies=["raw"],
        description="Normalize to zero mean, unit variance",
        post_funcs=[pval.validate_all_finite()],
    )

    # Extract subset
    pipeline.add_node(
        "subset",
        lambda normalized: normalized.isel(x=slice(0, 2), z=slice(0, 3)),
        dependencies=["normalized"],
        description="Extract subset",
    )

    # Execute
    results = pipeline.execute(outputs=["subset"])
    print(f"Result shape: {results['subset'].shape}")
    print(f"Result dims: {results['subset'].dims}")
    print()


def demo_conditional_pipeline():
    """Demonstrate conditional pipeline construction."""
    print("=" * 60)
    print("Demo 3: Conditional Pipeline Construction")
    print("=" * 60)

    def build_pipeline(mode: str) -> Pipeline:
        pipeline = Pipeline()

        pipeline.add_node("data", lambda: np.array([1, 2, 3, 4, 5]))

        if mode == "simple":
            pipeline.add_node(
                "result",
                lambda data: data * 2,
                dependencies=["data"],
                description="Double the data",
                metadata={"final": "true"},
            )
        elif mode == "complex":
            pipeline.add_node(
                "intermediate",
                lambda data: data * 2,
                dependencies=["data"],
                description="Double the data",
            )
            pipeline.add_node(
                "result",
                lambda intermediate: intermediate + 10,
                dependencies=["intermediate"],
                description="Add 10",
                metadata={"final": "true"},
            )

        return pipeline

    # Build and execute simple pipeline
    simple = build_pipeline("simple")
    simple_results = simple.execute(outputs=["result"])
    print(f"Simple mode result: {simple_results['result']}")

    # Build and execute complex pipeline
    complex_pipeline = build_pipeline("complex")
    complex_results = complex_pipeline.execute(outputs=["result"])
    print(f"Complex mode result: {complex_results['result']}")
    print()


def demo_bypassing():
    """Demonstrate data bypassing."""
    print("=" * 60)
    print("Demo 4: Data Bypassing")
    print("=" * 60)

    pipeline = Pipeline()

    # Simulate expensive computation
    def expensive_computation():
        print("  Running expensive computation...")
        return np.array([1, 2, 3, 4, 5])

    pipeline.add_node("expensive", expensive_computation, description="Expensive step")
    pipeline.add_node(
        "processed",
        lambda expensive: expensive * 2,
        dependencies=["expensive"],
        description="Process data",
    )
    pipeline.add_node(
        "final",
        lambda processed: processed + 10,
        dependencies=["processed"],
        description="Final step",
    )

    # Execute normally
    print("Normal execution:")
    results1 = pipeline.execute(outputs=["final"])
    print(f"Result: {results1['final']}\n")

    # Execute with bypassing
    print("Execution with bypassing (skips expensive computation):")
    cached_data = np.array([10, 20, 30, 40, 50])
    results2 = pipeline.execute(outputs=["final"], inputs={"expensive": cached_data})
    print(f"Result: {results2['final']}")
    print()


def demo_intermediate_outputs():
    """Demonstrate inspecting intermediate values."""
    print("=" * 60)
    print("Demo 5: Intermediate Outputs (Debugging)")
    print("=" * 60)

    pipeline = Pipeline()

    pipeline.add_node("step1", lambda: np.array([1, 2, 3]))
    pipeline.add_node("step2", lambda step1: step1 * 2, dependencies=["step1"])
    pipeline.add_node("step3", lambda step2: step2 + 10, dependencies=["step2"])

    # Request intermediate values alongside the final output
    results = pipeline.execute(outputs=["step3", "step1", "step2"])

    print("All values (including intermediates):")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print()


def demo_subgraph():
    """Demonstrate subgraph extraction."""
    print("=" * 60)
    print("Demo 6: Subgraph Extraction")
    print("=" * 60)

    # Build full pipeline with multiple branches
    pipeline = Pipeline()

    pipeline.add_node("data", lambda: np.array([1, 2, 3]))

    # Branch 1
    pipeline.add_node("double", lambda data: data * 2, dependencies=["data"])
    pipeline.add_node("double_plus", lambda double: double + 1, dependencies=["double"])

    # Branch 2
    pipeline.add_node("triple", lambda data: data * 3, dependencies=["data"])
    pipeline.add_node("triple_plus", lambda triple: triple + 1, dependencies=["triple"])

    print(f"Full pipeline nodes: {pipeline.list_nodes()}")

    # Extract subgraph for only branch 1
    subgraph = pipeline.extract_subgraph(["double_plus"])
    print(f"Subgraph nodes: {subgraph.list_nodes()}")

    # Execute subgraph
    results = subgraph.execute(outputs=["double_plus"])
    print(f"Subgraph result: {results['double_plus']}")
    print()


def demo_split_node():
    """Demonstrate add_split_node for dict-returning functions."""
    print("=" * 60)
    print("Demo 7: Split Nodes (add_split_node)")
    print("=" * 60)

    # A realistic scenario: a single function computes several related
    # quantities at once (e.g. to share an expensive intermediate result).
    # add_split_node exposes each value as an independent pipeline node so
    # that downstream nodes can depend on individual outputs.

    def compute_statistics(data):
        """Compute multiple statistics in one pass."""
        return {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }

    pipeline = Pipeline()

    pipeline.add_node("data", lambda: np.arange(10, dtype=float))

    # The source node "_stats" holds the full dict; mean/std/min/max are
    # independent child nodes that downstream nodes can depend on separately.
    pipeline.add_node(
        "_stats",
        compute_statistics,
        outputs={
            "mean": lambda d: d["mean"],
            "std": lambda d: d["std"],
            "min": lambda d: d["min"],
            "max": lambda d: d["max"],
        },
        dependencies=["data"],
        description="Compute summary statistics",
    )

    # A downstream node that only needs mean and std
    pipeline.add_node(
        "coefficient_of_variation",
        lambda mean, std: std / mean,
        dependencies=["mean", "std"],
        description="CV = std / mean",
    )

    results = pipeline.execute(
        outputs=["mean", "std", "min", "max", "coefficient_of_variation"]
    )
    print(f"  mean = {results['mean']:.2f}")
    print(f"  std  = {results['std']:.2f}")
    print(f"  min  = {results['min']:.2f}")
    print(f"  max  = {results['max']:.2f}")
    print(f"  CV   = {results['coefficient_of_variation']:.4f}")
    print()


def demo_get_nodes_by_metadata():
    """Demonstrate get_nodes_by_metadata for dynamic output discovery."""
    print("=" * 60)
    print("Demo 8: Metadata-Based Output Discovery (get_nodes_by_metadata)")
    print("=" * 60)

    # Tag nodes with metadata to mark which outputs are "final" products.
    # get_nodes_by_metadata() then lets the caller discover those nodes
    # without hardcoding their names — useful when the set of outputs is
    # determined at pipeline-build time (e.g. conditioned on a config flag).

    def build_pipeline(compute_ratio: bool) -> Pipeline:
        pipeline = Pipeline()

        pipeline.add_node("data", lambda: np.array([1.0, 4.0, 9.0, 16.0]))

        pipeline.add_node(
            "sqrt_data",
            lambda data: np.sqrt(data),
            dependencies=["data"],
            description="Square root",
            metadata={"final": True, "kind": "data"},
        )
        pipeline.add_node(
            "log_data",
            lambda data: np.log(data),
            dependencies=["data"],
            description="Natural log",
            metadata={"final": True, "kind": "data"},
        )

        if compute_ratio:
            pipeline.add_node(
                "ratio",
                lambda sqrt_data, log_data: sqrt_data / log_data,
                dependencies=["sqrt_data", "log_data"],
                description="sqrt / log",
                metadata={"final": True, "kind": "data"},
            )

        return pipeline

    for flag in [False, True]:
        pipeline = build_pipeline(compute_ratio=flag)

        # Discover outputs dynamically — no hardcoded list
        output_names = pipeline.get_nodes_by_metadata(final=True, kind="data")
        results = pipeline.execute(outputs=output_names)

        print(f"  compute_ratio={flag}  →  outputs: {output_names}")
        for name, value in results.items():
            print(f"    {name}: {np.round(value, 3)}")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("*" * 60)
    print("Pipeline Engine Demonstration")
    print("*" * 60)
    print("\n")

    demo_basic_pipeline()
    demo_xarray_pipeline()
    demo_conditional_pipeline()
    demo_bypassing()
    demo_intermediate_outputs()
    demo_subgraph()
    demo_split_node()
    demo_get_nodes_by_metadata()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
