"""Demonstration of pipeline visualization.

This script shows how to visualize pipelines using text summaries
and export to various formats.
"""

import numpy as np

from eradiate.pipelines import Pipeline


def build_example_pipeline():
    """Build an example postprocessing pipeline."""
    pipeline = Pipeline()

    # Raw data node
    pipeline.add_node(
        "raw_data",
        func=lambda: np.random.randn(10, 5, 3),
        description="Load raw data from solver",
    )

    # Reshape
    pipeline.add_node(
        "reshaped",
        func=lambda raw_data: raw_data.reshape(-1, 3),
        dependencies=["raw_data"],
        description="Reshape to 2D array",
    )

    # Statistics branch
    pipeline.add_node(
        "mean",
        func=lambda reshaped: np.mean(reshaped, axis=0),
        dependencies=["reshaped"],
        description="Compute mean along axis 0",
        metadata={"final": "true", "kind": "statistics"},
    )

    pipeline.add_node(
        "std",
        func=lambda reshaped: np.std(reshaped, axis=0),
        dependencies=["reshaped"],
        description="Compute standard deviation",
        metadata={"final": "true", "kind": "statistics"},
    )

    # Normalization branch
    pipeline.add_node(
        "normalized",
        func=lambda reshaped: (reshaped - np.mean(reshaped, axis=0))
        / np.std(reshaped, axis=0),
        dependencies=["reshaped"],
        description="Normalize to zero mean, unit std",
    )

    pipeline.add_node(
        "result",
        func=lambda normalized: normalized.reshape(10, 5, 3),
        dependencies=["normalized"],
        description="Reshape back to original shape",
        metadata={"final": "true", "kind": "data"},
    )

    return pipeline


def main():
    """Run visualization demo."""
    print("\n" + "=" * 70)
    print("Pipeline Visualization Demo")
    print("=" * 70 + "\n")

    # Build pipeline
    pipeline = build_example_pipeline()

    # Text summary
    print("\n1. Text Summary")
    print("-" * 70)
    pipeline.print_summary()

    # Graphviz export (if pydot is available)
    print("\n2. Graphviz DOT Export")
    print("-" * 70)
    try:
        dot_file = "examples/pipeline.dot"
        pipeline.write_dot(dot_file)
        print(f"Exported Graphviz DOT to: {dot_file}")
        print("\nTo render the visualization:")
        print(f"  dot -Tpng {dot_file} -o pipeline.png")
        print(f"  dot -Tsvg {dot_file} -o pipeline.svg")
    except ImportError:
        print(
            "pydot not installed. Install with: uv add --optional visualization pydot"
        )
        print("Or: pip install pydot")

    print("\n" + "=" * 70)
    print("Visualization demo completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
