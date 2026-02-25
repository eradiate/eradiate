"""Tests for virtual inputs in the pipeline engine."""

import pytest

from eradiate.pipelines import Pipeline


class TestBasicVirtualInputs:
    """Test basic virtual input functionality."""

    def test_single_virtual_input_single_consumer(self):
        """Test a single virtual input consumed by one node."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        # Check virtual input is tracked
        assert pipeline.get_virtual_inputs() == ["a"]
        assert pipeline.is_virtual_input("a")

        # Execute with virtual input provided
        results = pipeline.execute(outputs=["b"], inputs={"a": 10})
        assert results["b"] == 11

    def test_multiple_nodes_same_virtual_input(self):
        """Test multiple nodes depending on the same virtual input."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda a: a * 2, dependencies=["a"])

        # Virtual input 'a' should appear once
        assert pipeline.get_virtual_inputs() == ["a"]

        # Execute both outputs
        results = pipeline.execute(outputs=["b", "c"], inputs={"a": 5})
        assert results["b"] == 6
        assert results["c"] == 10

    def test_execute_without_virtual_input_raises_error(self):
        """Test that missing virtual inputs raise an error."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        with pytest.raises(ValueError, match="Missing required virtual inputs"):
            pipeline.execute(outputs=["b"])

    def test_multiple_virtual_inputs(self):
        """Test multiple virtual inputs in dependency chain."""
        pipeline = Pipeline()
        pipeline.add_node("c", lambda a, b: a + b, dependencies=["a", "b"])

        assert set(pipeline.get_virtual_inputs()) == {"a", "b"}

        results = pipeline.execute(outputs=["c"], inputs={"a": 3, "b": 7})
        assert results["c"] == 10


class TestIntrospectionMethods:
    """Test introspection methods for virtual inputs."""

    def test_get_virtual_inputs_empty(self):
        """Test get_virtual_inputs on pipeline with no virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        assert pipeline.get_virtual_inputs() == []

    def test_get_required_inputs_for_outputs(self):
        """Test get_required_inputs for specific outputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])

        # For output 'c', need virtual input 'a'
        assert pipeline.get_required_inputs(outputs=["c"]) == ["a"]

        # For output 'b', also need 'a'
        assert pipeline.get_required_inputs(outputs=["b"]) == ["a"]

    def test_get_required_inputs_with_bypass(self):
        """Test that bypassing nodes reduces required inputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])

        # Bypassing 'b' means we don't need 'a'
        assert pipeline.get_required_inputs(outputs=["c"], inputs={"b": 5}) == []

    def test_get_required_inputs_default_outputs(self):
        """Test get_required_inputs with default outputs (leaf nodes)."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        # Default to leaf nodes ('b' in this case)
        assert pipeline.get_required_inputs() == ["a"]

    def test_is_virtual_input_false_for_node(self):
        """Test is_virtual_input returns False for regular nodes."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda x: x + 1, dependencies=["x"])

        assert not pipeline.is_virtual_input("a")
        assert pipeline.is_virtual_input("x")


class TestVirtualInputBecomesRealNode:
    """Test behavior when a virtual input is later added as a real node."""

    def test_virtual_input_becomes_real_node(self):
        """Test adding a node with the name of a virtual input."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        # 'a' is initially virtual
        assert "a" in pipeline.get_virtual_inputs()

        # Add 'a' as a real node
        pipeline.add_node("a", lambda: 10)

        # 'a' should no longer be virtual
        assert "a" not in pipeline.get_virtual_inputs()

        # Execute should work without providing 'a'
        results = pipeline.execute(outputs=["b"])
        assert results["b"] == 11


class TestConnectivityValidation:
    """Test connectivity validation for virtual inputs."""

    def test_disconnected_output_raises_error(self):
        """Test that outputs unreachable from inputs raise an error."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda x: x * 2, dependencies=["x"])

        # Only provide 'a', not 'x' - should error about missing 'x'
        with pytest.raises(ValueError, match="Missing required virtual inputs"):
            pipeline.execute(outputs=["c"], inputs={"a": 10})

    def test_partially_provided_virtual_inputs_error(self):
        """Test error when some but not all required virtual inputs provided."""
        pipeline = Pipeline()
        pipeline.add_node("d", lambda a, b, c: a + b + c, dependencies=["a", "b", "c"])

        # Only provide 'a' and 'b', missing 'c'
        with pytest.raises(
            ValueError, match="Missing required virtual inputs: \\['c'\\]"
        ):
            pipeline.execute(outputs=["d"], inputs={"a": 1, "b": 2})


class TestMixedBypassData:
    """Test inputs containing both node bypasses and virtual inputs."""

    def test_bypass_node_and_virtual_input(self):
        """Test providing both node bypass and virtual input in inputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])

        # Bypass node 'b' and provide virtual input 'a'
        results = pipeline.execute(outputs=["c"], inputs={"b": 5, "a": 10})
        assert results["c"] == 10  # c = b * 2 = 5 * 2

    def test_invalid_bypass_key_raises_error(self):
        """Test that invalid keys in inputs raise an error."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        with pytest.raises(ValueError, match="neither a node nor a virtual input"):
            pipeline.execute(outputs=["b"], inputs={"a": 10, "invalid": 99})


class TestSubgraphExtraction:
    """Test subgraph extraction with virtual inputs."""

    def test_extract_subgraph_preserves_virtual_inputs(self):
        """Test that extracting a subgraph preserves required virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])
        pipeline.add_node("d", lambda x: x - 1, dependencies=["x"])

        # Extract subgraph for 'c'
        subgraph = pipeline.extract_subgraph(outputs=["c"])

        # Should have virtual input 'a' but not 'x'
        assert "a" in subgraph.get_virtual_inputs()
        assert "x" not in subgraph.get_virtual_inputs()

        # Should be able to execute with just 'a'
        results = subgraph.execute(outputs=["c"], inputs={"a": 5})
        assert results["c"] == 12  # c = (a + 1) * 2 = (5 + 1) * 2

    def test_extract_subgraph_without_virtual_inputs(self):
        """Test extracting subgraph from nodes with no virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda x: x * 2, dependencies=["x"])

        # Extract subgraph for 'b' (no virtual inputs)
        subgraph = pipeline.extract_subgraph(outputs=["b"])

        assert subgraph.get_virtual_inputs() == []
        results = subgraph.execute(outputs=["b"])
        assert results["b"] == 2


class TestNodeRemoval:
    """Test node removal with virtual inputs."""

    def test_remove_node_cleans_up_orphaned_virtual_input(self):
        """Test that removing a node cleans up orphaned virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        assert "a" in pipeline.get_virtual_inputs()

        # Remove 'b', making 'a' orphaned
        pipeline.remove_node("b")

        # 'a' should be cleaned up
        assert "a" not in pipeline.get_virtual_inputs()

    def test_remove_node_preserves_shared_virtual_input(self):
        """Test that removing a node doesn't remove shared virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda a: a * 2, dependencies=["a"])

        assert "a" in pipeline.get_virtual_inputs()

        # Remove 'b', but 'a' is still used by 'c'
        pipeline.remove_node("b")

        # 'a' should still be virtual
        assert "a" in pipeline.get_virtual_inputs()


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_pipeline_virtual_inputs(self):
        """Test get_virtual_inputs on empty pipeline."""
        pipeline = Pipeline()
        assert pipeline.get_virtual_inputs() == []

    def test_cycle_with_virtual_input_raises_error(self):
        """Test that cycles involving virtual inputs are detected."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        # Try to add 'a' depending on 'b', creating a cycle
        with pytest.raises(ValueError, match="would create a cycle"):
            pipeline.add_node("a", lambda b: b * 2, dependencies=["b"])

    def test_multiple_virtual_inputs_dependency_chain(self):
        """Test multiple virtual inputs in a dependency chain."""
        pipeline = Pipeline()
        pipeline.add_node("c", lambda a, b: a + b, dependencies=["a", "b"])
        pipeline.add_node("d", lambda c: c * 2, dependencies=["c"])
        pipeline.add_node("e", lambda d, x: d + x, dependencies=["d", "x"])

        # Should have virtual inputs 'a', 'b', 'x'
        assert set(pipeline.get_virtual_inputs()) == {"a", "b", "x"}

        # Execute with all virtual inputs
        results = pipeline.execute(outputs=["e"], inputs={"a": 1, "b": 2, "x": 5})
        assert results["e"] == 11  # e = ((a + b) * 2) + x = ((1 + 2) * 2) + 5

    def test_node_with_no_dependencies(self):
        """Test that nodes with no dependencies don't create virtual inputs."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 42)

        assert pipeline.get_virtual_inputs() == []

    def test_virtual_input_in_bypassed_branch(self):
        """Test virtual input in a branch that's bypassed."""
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])

        # Bypass 'b', so 'a' is not required
        results = pipeline.execute(outputs=["c"], inputs={"b": 10})
        assert results["c"] == 20

        # Should not need to provide 'a'
        required = pipeline.get_required_inputs(outputs=["c"], inputs={"b": 10})
        assert "a" not in required
