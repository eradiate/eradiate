"""Tests for core pipeline functionality."""

import pytest

from eradiate.pipelines.engine import Pipeline

# ------------------------------------------------------------------------------
#                                 Shared fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def pipeline_ab():
    """Linear two-node pipeline: a → b (a=1, b=2)."""
    p = Pipeline()
    p.add_node("a", lambda: 1)
    p.add_node("b", lambda a: a + 1, dependencies=["a"])
    return p


@pytest.fixture
def pipeline_abc():
    """Linear three-node chain: a → b → c (a=1, b=2, c=3)."""
    p = Pipeline()
    p.add_node("a", lambda: 1)
    p.add_node("b", lambda a: a + 1, dependencies=["a"])
    p.add_node("c", lambda b: b + 1, dependencies=["b"])
    return p


@pytest.fixture
def collector():
    """Return (recorded_list, recorder_func) for tracking validator calls."""
    recorded = []

    def record(value):
        recorded.append(value)

    return recorded, record


# ------------------------------------------------------------------------------


class TestPipelineBasics:
    """Tests for basic pipeline operations."""

    def test_add_single_node(self):
        """Test adding a single node."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        assert "a" in pipeline._nodes
        assert len(pipeline._nodes) == 1

    def test_add_node_with_dependencies(self, pipeline_ab):
        """Test adding nodes with dependencies."""
        assert "b" in pipeline_ab._nodes
        assert pipeline_ab._nodes["b"].dependencies == ["a"]

    def test_add_node_chaining(self):
        """Test method chaining for add_node."""
        pipeline = Pipeline()
        result = (
            pipeline.add_node("a", lambda: 1)
            .add_node("b", lambda: 2)
            .add_node("c", lambda a, b: a + b, dependencies=["a", "b"])
        )
        assert result is pipeline
        assert len(pipeline._nodes) == 3

    def test_add_duplicate_node_raises(self):
        """Test that adding duplicate node raises error."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_node("a", lambda: 2)

    def test_add_node_missing_dependency_creates_virtual_input(self):
        """Test that missing dependency creates a virtual input."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda b: b + 1, dependencies=["b"])
        # 'b' should be tracked as a virtual input
        assert "b" in pipeline.get_virtual_inputs()

    def test_add_node_with_description(self):
        """Test adding node with description."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, description="Returns one")
        assert pipeline._nodes["a"].description == "Returns one"

    def test_add_node_with_metadata(self):
        """Test adding node with metadata."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"final": "true", "kind": "data"})
        assert pipeline._nodes["a"].metadata == {"final": "true", "kind": "data"}


class TestPipelineCycleDetection:
    """Tests for cycle detection."""

    def test_simple_cycle_raises(self):
        """Test that a two-node cycle (a→b→a) is detected."""
        # Add b depending on a (a becomes a virtual input), then try to close
        # the cycle by adding a depending on b.
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        with pytest.raises(ValueError, match="would create a cycle"):
            pipeline.add_node("a", lambda b: b + 1, dependencies=["b"])

    def test_self_dependency_raises(self):
        """Test that a self-loop (a→a) is detected."""
        # When adding "a" with itself as a dependency, "a" is first treated as
        # a virtual input, then the node is added with an a→a edge which forms
        # a self-loop.
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="would create a cycle"):
            pipeline.add_node("a", lambda a: a + 1, dependencies=["a"])

    def test_indirect_cycle_raises(self):
        """Test that an indirect cycle (a→b→c→a) is detected."""
        # Build the chain b→c first (making a a virtual input), then try to
        # add a depending on c, which would close the cycle.
        pipeline = Pipeline()
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda b: b + 1, dependencies=["b"])
        with pytest.raises(ValueError, match="would create a cycle"):
            pipeline.add_node("a", lambda c: c + 1, dependencies=["c"])


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_execute_single_node(self):
        """Test executing single node."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 42)
        result = pipeline.execute(outputs=["a"])
        assert result == {"a": 42}

    def test_execute_with_dependencies(self):
        """Test executing nodes with dependencies."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda: 2)
        pipeline.add_node("c", lambda a, b: a + b, dependencies=["a", "b"])
        result = pipeline.execute(outputs=["c"])
        assert result == {"c": 3}

    def test_execute_multiple_outputs(self):
        """Test executing multiple outputs."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a * 2, dependencies=["a"])
        pipeline.add_node("c", lambda a: a * 3, dependencies=["a"])
        result = pipeline.execute(outputs=["b", "c"])
        assert result == {"b": 2, "c": 3}

    def test_execute_caches_results(self):
        """Test that execution caches results."""
        call_count = 0

        def expensive_func():
            nonlocal call_count
            call_count += 1
            return 42

        pipeline = Pipeline()
        pipeline.add_node("expensive", expensive_func)
        pipeline.add_node(
            "b", lambda expensive: expensive + 1, dependencies=["expensive"]
        )
        pipeline.add_node(
            "c", lambda expensive: expensive + 2, dependencies=["expensive"]
        )

        result = pipeline.execute(outputs=["b", "c"])
        assert result == {"b": 43, "c": 44}
        # Should only call expensive_func once
        assert call_count == 1

    def test_execute_lazy_evaluation(self):
        """Test that execution only computes required nodes."""
        executed = []

        def track_execution(name):
            def func():
                executed.append(name)
                return name

            return func

        pipeline = Pipeline()
        pipeline.add_node("a", track_execution("a"))
        pipeline.add_node("b", track_execution("b"))
        pipeline.add_node("c", lambda a: a, dependencies=["a"])

        # Only execute 'c', should not execute 'b'
        pipeline.execute(outputs=["c"])
        assert "a" in executed
        assert "c" not in executed  # c uses a's value directly
        assert "b" not in executed

    def test_execute_no_outputs_computes_leaves(self):
        """Test that execute with no outputs computes leaf nodes."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda a: a + 2, dependencies=["a"])

        result = pipeline.execute()  # No outputs specified
        # Should compute all leaf nodes (b and c)
        assert set(result.keys()) == {"b", "c"}
        assert result["b"] == 2
        assert result["c"] == 3

    def test_execute_nonexistent_output_raises(self):
        """Test that requesting nonexistent output raises error."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        with pytest.raises(ValueError, match="not found"):
            pipeline.execute(outputs=["nonexistent"])


class TestPipelineBypass:
    """Tests for data bypassing."""

    def test_bypass_single_node(self, pipeline_ab):
        """Test bypassing a single node."""
        result = pipeline_ab.execute(outputs=["b"], inputs={"a": 10})
        assert result == {"b": 11}

    def test_bypass_skips_computation(self):
        """Test that bypass skips node computation."""
        executed = []

        def track_execution():
            executed.append("a")
            return 1

        pipeline = Pipeline()
        pipeline.add_node("a", track_execution)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        pipeline.execute(outputs=["b"], inputs={"a": 10})
        assert "a" not in executed  # Should not execute 'a'

    def test_bypass_intermediate_node(self):
        """Test bypassing intermediate node."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a * 2, dependencies=["a"])
        pipeline.add_node("c", lambda b: b + 1, dependencies=["b"])

        result = pipeline.execute(outputs=["c"], inputs={"b": 10})
        assert result == {"c": 11}


class TestPipelineSubgraph:
    """Tests for subgraph extraction."""

    def test_extract_simple_subgraph(self):
        """Test extracting simple subgraph."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda: 2)

        subgraph = pipeline.extract_subgraph(["b"])
        assert set(subgraph._nodes.keys()) == {"a", "b"}
        assert "c" not in subgraph._nodes

    def test_extract_subgraph_multiple_outputs(self):
        """Test extracting subgraph with multiple outputs."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        pipeline.add_node("c", lambda a: a + 2, dependencies=["a"])
        pipeline.add_node("d", lambda: 3)

        subgraph = pipeline.extract_subgraph(["b", "c"])
        assert set(subgraph._nodes.keys()) == {"a", "b", "c"}
        assert "d" not in subgraph._nodes

    def test_extract_subgraph_executes_correctly(self, pipeline_abc):
        """Test that extracted subgraph executes correctly."""
        subgraph = pipeline_abc.extract_subgraph(["b"])
        result = subgraph.execute(outputs=["b"])
        assert result == {"b": 2}

    def test_extract_subgraph_nonexistent_output_raises(self):
        """Test that extracting nonexistent output raises error."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        with pytest.raises(ValueError, match="not found"):
            pipeline.extract_subgraph(["nonexistent"])


class TestPipelineIntermediateOutputs:
    """Tests for requesting intermediate nodes via outputs."""

    def test_intermediate_and_leaf_outputs(self, pipeline_abc):
        """Test that intermediate nodes can be requested alongside leaf nodes."""
        result = pipeline_abc.execute(outputs=["c", "a", "b"])
        assert result == {"c": 3, "a": 1, "b": 2}

    def test_intermediate_only_output(self):
        """Test requesting only an intermediate node as output."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 10)
        pipeline.add_node("b", lambda a: a * 2, dependencies=["a"])
        pipeline.add_node("c", lambda b: b + 5, dependencies=["b"])

        result = pipeline.execute(outputs=["b"])
        assert result == {"b": 20}

    def test_intermediate_output_with_bypass(self, pipeline_abc):
        """Test intermediate outputs work alongside inputs."""
        result = pipeline_abc.execute(outputs=["c", "b"], inputs={"a": 10})
        assert result == {"c": 12, "b": 11}

    def test_intermediate_does_not_execute_descendants(self):
        """Test that requesting an intermediate doesn't execute its descendants."""
        executed = []

        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node(
            "b", lambda a: (executed.append("b"), a + 1)[1], dependencies=["a"]
        )
        pipeline.add_node(
            "c", lambda b: (executed.append("c"), b + 1)[1], dependencies=["b"]
        )

        result = pipeline.execute(outputs=["b"])
        assert result == {"b": 2}
        assert "b" in executed
        assert "c" not in executed


class TestPipelineRemoval:
    """Tests for node removal."""

    def test_remove_node(self):
        """Test removing a node."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda: 2)

        pipeline.remove_node("b")
        assert "b" not in pipeline._nodes
        assert "a" in pipeline._nodes

    def test_remove_node_with_dependents_raises(self, pipeline_ab):
        """Test that removing node with dependents raises error."""
        with pytest.raises(ValueError, match="depend on it"):
            pipeline_ab.remove_node("a")

    def test_remove_nonexistent_node_raises(self):
        """Test that removing nonexistent node raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.remove_node("nonexistent")


class TestPipelineUtilities:
    """Tests for utility methods."""

    def test_list_nodes(self, pipeline_abc):
        """Test listing nodes in topological order."""
        assert pipeline_abc.list_nodes() == ["a", "b", "c"]

    def test_get_node(self):
        """Test getting a node."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, description="Test")

        node = pipeline.get_node("a")
        assert node.name == "a"
        assert node.description == "Test"

    def test_get_nonexistent_node_raises(self):
        """Test that getting nonexistent node raises error."""
        pipeline = Pipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.get_node("nonexistent")

    def test_clear_cache(self):
        """Test clearing the cache."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)

        # Execute to populate cache
        pipeline.execute(outputs=["a"])
        assert "a" in pipeline._cache

        # Clear cache
        pipeline.clear_cache()
        assert len(pipeline._cache) == 0


class TestPipelineValidation:
    """Tests for validation functionality."""

    def test_post_validator_called(self, collector):
        """Test that post-validator is called."""
        validated, validator = collector
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 42, post_funcs=[validator])
        pipeline.execute(outputs=["a"])
        assert validated == [42]

    def test_post_validator_raises(self):
        """Test that post-validator can raise error."""

        def validator(value):
            if value < 0:
                raise ValueError("Must be positive")

        pipeline = Pipeline()
        pipeline.add_node("a", lambda: -1, post_funcs=[validator])

        with pytest.raises(ValueError, match="Must be positive"):
            pipeline.execute(outputs=["a"])

    def test_pre_validator_called(self, collector):
        """Test that pre-validator is called."""
        validated, validator = collector
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node(
            "b", lambda a: a + 1, dependencies=["a"], pre_funcs=[validator]
        )
        pipeline.execute(outputs=["b"])
        assert len(validated) == 1
        assert validated[0] == {"a": 1}

    def test_validation_disabled_locally(self, collector):
        """Test that validation can be disabled per-node."""
        validated, validator = collector
        pipeline = Pipeline(validate=True)
        pipeline.add_node("a", lambda: 42, post_funcs=[validator], validate=False)
        pipeline.execute(outputs=["a"])
        assert validated == []

    def test_validation_disabled_globally(self, collector):
        """Test that validation can be disabled globally."""
        validated, validator = collector
        pipeline = Pipeline(validate=False)
        pipeline.add_node("a", lambda: 42, post_funcs=[validator])
        pipeline.execute(outputs=["a"])
        assert validated == []

    def test_multiple_validators(self):
        """Test multiple validators on same node."""
        validated = []

        def validator1(value):
            validated.append(("v1", value))

        def validator2(value):
            validated.append(("v2", value))

        pipeline = Pipeline()
        pipeline.add_node(
            "a",
            lambda: 42,
            post_funcs=[validator1, validator2],
        )
        pipeline.execute(outputs=["a"])

        assert validated == [("v1", 42), ("v2", 42)]


class TestAddNodeOutputs:
    """Tests for add_node() with the outputs parameter (field extraction)."""

    def test_adds_source_and_child_nodes(self):
        """Test that source node and all child nodes are registered."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"x": 1, "y": 2},
            outputs={"x": lambda d: d["x"], "y": lambda d: d["y"]},
        )
        assert "_raw" in pipeline._nodes
        assert "x" in pipeline._nodes
        assert "y" in pipeline._nodes

    def test_execution_produces_correct_values(self):
        """Test that each child node extracts the correct value from the dict."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"x": 10, "y": 20},
            outputs={"x": lambda d: d["x"], "y": lambda d: d["y"]},
        )
        result = pipeline.execute(outputs=["x", "y"])
        assert result == {"x": 10, "y": 20}

    def test_with_upstream_dependencies(self):
        """Test outputs= with upstream node dependencies."""
        pipeline = Pipeline()
        pipeline.add_node("scale", lambda: 3)
        pipeline.add_node(
            "_raw",
            lambda scale: {"a": scale * 1, "b": scale * 2},
            outputs={"a": lambda d: d["a"], "b": lambda d: d["b"]},
            dependencies=["scale"],
        )
        result = pipeline.execute(outputs=["a", "b"])
        assert result == {"a": 3, "b": 6}

    def test_source_func_called_once(self):
        """Test that the source function is called only once even with multiple outputs."""
        call_count = 0

        def source():
            nonlocal call_count
            call_count += 1
            return {"x": 1, "y": 2}

        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            source,
            outputs={"x": lambda d: d["x"], "y": lambda d: d["y"]},
        )
        pipeline.execute(outputs=["x", "y"])
        assert call_count == 1

    def test_method_chaining(self):
        """Test that add_node with outputs= returns self for chaining."""
        pipeline = Pipeline()
        result = pipeline.add_node(
            "_raw",
            lambda: {"x": 1},
            outputs={"x": lambda d: d["x"]},
        )
        assert result is pipeline

    def test_metadata_attached_to_source_node(self):
        """Test that metadata is attached to the source node, not child nodes."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"x": 1},
            outputs={"x": lambda d: d["x"]},
            metadata={"kind": "raw"},
        )
        assert pipeline._nodes["_raw"].metadata == {"kind": "raw"}
        assert pipeline._nodes["x"].metadata == {}

    def test_child_depends_on_source(self):
        """Test that each child node lists the source as its dependency."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"x": 1, "y": 2},
            outputs={"x": lambda d: d["x"], "y": lambda d: d["y"]},
        )
        assert pipeline._nodes["x"].dependencies == ["_raw"]
        assert pipeline._nodes["y"].dependencies == ["_raw"]

    def test_with_virtual_input_dependency(self):
        """Test outputs= when dependencies include a virtual input."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda data: {"a": data[0], "b": data[1]},
            outputs={"a": lambda d: d["a"], "b": lambda d: d["b"]},
            dependencies=["data"],
        )
        assert "data" in pipeline.get_virtual_inputs()
        result = pipeline.execute(outputs=["a", "b"], inputs={"data": [7, 8]})
        assert result == {"a": 7, "b": 8}

    def test_outputs_list_form(self):
        """Test outputs= as list[str]: node ID equals dict key."""
        pipeline = Pipeline()
        pipeline.add_node("_raw", lambda: {"x": 10, "y": 20}, outputs=["x", "y"])
        assert "x" in pipeline._nodes
        assert "y" in pipeline._nodes
        result = pipeline.execute(outputs=["x", "y"])
        assert result == {"x": 10, "y": 20}

    def test_outputs_dict_str_str_form(self):
        """Test outputs= as dict[str, str]: node ID mapped to dict key."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"x_internal": 1, "y_internal": 2},
            outputs={"x": "x_internal", "y": "y_internal"},
        )
        assert "x" in pipeline._nodes
        assert "y" in pipeline._nodes
        result = pipeline.execute(outputs=["x", "y"])
        assert result == {"x": 1, "y": 2}

    def test_outputs_dict_mixed_str_callable(self):
        """Test outputs= as dict mixing str and Callable values."""
        pipeline = Pipeline()
        pipeline.add_node(
            "_raw",
            lambda: {"a": 3, "b": 4},
            outputs={"a": "a", "sum": lambda d: d["a"] + d["b"]},
        )
        result = pipeline.execute(outputs=["a", "sum"])
        assert result == {"a": 3, "sum": 7}


class TestPipelineGetNodesByMetadata:
    """Tests for Pipeline.get_nodes_by_metadata()."""

    def test_single_key_match(self):
        """Test filtering nodes by a single metadata key."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"final": True})
        pipeline.add_node("b", lambda: 2, metadata={"final": False})
        pipeline.add_node("c", lambda: 3)

        result = pipeline.get_nodes_by_metadata(final=True)
        assert result == ["a"]

    def test_multiple_keys_all_must_match(self):
        """Test that all specified key-value pairs must be present."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"final": True, "kind": "data"})
        pipeline.add_node("b", lambda: 2, metadata={"final": True, "kind": "debug"})
        pipeline.add_node("c", lambda: 3, metadata={"final": True})

        result = pipeline.get_nodes_by_metadata(final=True, kind="data")
        assert result == ["a"]

    def test_returns_topological_order(self):
        """Test that matched nodes are returned in topological order."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"tag": "x"})
        pipeline.add_node(
            "b", lambda a: a + 1, dependencies=["a"], metadata={"tag": "x"}
        )
        pipeline.add_node(
            "c", lambda b: b + 1, dependencies=["b"], metadata={"tag": "x"}
        )

        result = pipeline.get_nodes_by_metadata(tag="x")
        assert result == ["a", "b", "c"]

    def test_excludes_virtual_inputs(self):
        """Test that virtual inputs are never returned."""
        pipeline = Pipeline()
        # 'a' becomes a virtual input, not a node
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        # No kwargs → vacuously matches all nodes (but not virtual inputs)
        result = pipeline.get_nodes_by_metadata()
        assert "a" not in result
        assert "b" in result

    def test_empty_result_when_no_match(self):
        """Test that an empty list is returned when nothing matches."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"kind": "data"})

        result = pipeline.get_nodes_by_metadata(kind="other")
        assert result == []

    def test_no_kwargs_matches_all_real_nodes(self):
        """Test that calling with no kwargs returns all non-virtual-input nodes."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1)
        pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

        result = pipeline.get_nodes_by_metadata()
        assert set(result) == {"a", "b"}

    def test_partial_metadata_does_not_match(self):
        """Test that a node missing one of the specified keys is excluded."""
        pipeline = Pipeline()
        pipeline.add_node("a", lambda: 1, metadata={"final": True})  # no "kind" key

        result = pipeline.get_nodes_by_metadata(final=True, kind="data")
        assert result == []


class TestExecutionContext:
    """Tests for _execution_context (used by visualization methods)."""

    @pytest.fixture
    def pipeline_chain(self):
        """Linear chain: vi(a) -> b -> c -> d."""
        p = Pipeline()
        p.add_node("b", lambda a: a + 1, dependencies=["a"])
        p.add_node("c", lambda b: b * 2, dependencies=["b"])
        p.add_node("d", lambda c: c - 1, dependencies=["c"])
        return p

    @pytest.fixture
    def pipeline_diamond(self):
        """Diamond: vi(a) -> b, vi(a) -> c, b+c -> d."""
        p = Pipeline()
        p.add_node("b", lambda a: a + 1, dependencies=["a"])
        p.add_node("c", lambda a: a * 2, dependencies=["a"])
        p.add_node("d", lambda b, c: b + c, dependencies=["b", "c"])
        return p

    def test_full_pipeline_no_context(self, pipeline_chain):
        """Without outputs/inputs, all nodes are active and nothing is inactive."""
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=None, inputs=None
        )
        assert active == {"b", "c", "d"}
        assert bypassed == set()
        assert satisfied == set()
        assert inactive == set()

    def test_output_subset_makes_nodes_inactive(self, pipeline_chain):
        """Requesting only 'c' should mark 'd' as inactive."""
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=["c"], inputs=None
        )
        assert active == {"b", "c"}
        assert inactive == {"d"}

    def test_bypass_node_appears_in_bypassed(self, pipeline_chain):
        """Bypassing 'b' should put it in bypassed, not active."""
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=["d"], inputs={"b": 5}
        )
        assert "b" in bypassed
        assert "b" not in active
        assert "a" in inactive  # 'a' is upstream of bypass 'b' — no longer needed

    def test_satisfied_virtual_input(self, pipeline_chain):
        """Providing a virtual input value puts it in satisfied_vis."""
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=["d"], inputs={"a": 1}
        )
        assert "a" in satisfied
        assert "a" not in inactive

    def test_inactive_virtual_input(self, pipeline_chain):
        """Virtual inputs outside the requested subgraph appear as inactive."""
        # Add a second unrelated branch
        pipeline_chain.add_node("z", lambda x: x, dependencies=["x"])
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=["d"], inputs=None
        )
        assert "x" in inactive  # 'x' feeds only 'z', not needed for 'd'
        assert "z" in inactive

    def test_bypass_cuts_upstream_from_active(self, pipeline_chain):
        """When 'b' is bypassed, 'a' (its only supplier) becomes inactive."""
        active, bypassed, satisfied, inactive = pipeline_chain._execution_context(
            outputs=["d"], inputs={"b": 5}
        )
        # The backward BFS stops at bypassed 'b', so 'a' is outside the subgraph
        assert "a" in inactive
        assert "a" not in satisfied  # not provided

    def test_diamond_partial_bypass(self, pipeline_diamond):
        """Bypassing one branch of a diamond leaves the other active."""
        active, bypassed, satisfied, inactive = pipeline_diamond._execution_context(
            outputs=["d"], inputs={"b": 3}
        )
        assert "b" in bypassed
        assert "c" in active
        assert "d" in active
        assert inactive == set()
