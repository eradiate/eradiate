"""Core pipeline engine implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Optional

import attrs
import networkx as nx

_DOT_STYLES = {
    "default_node": {"fontname": "Helvetica", "fontsize": "10"},
    "default_edge": {"fontname": "Helvetica", "fontsize": "9"},
    "legend_box": {"style": "dashed", "color": "lightgrey"},
    "node_computation": {"style": "filled", "shape": "box", "fillcolor": "lightblue"},
    "node_virtual_input": {"shape": "ellipse", "fillcolor": "gold", "style": "filled"},
    "node_highlight": {"style": "filled", "fillcolor": "lightcoral"},
}


@attrs.define
class Node:
    """Represents a computation node in the pipeline.

    Parameters
    ----------
    name : str
        Unique identifier for the node.

    func : Callable
        Function to execute for this node. Parameters must match dependency names.

    dependencies : list of str
        Names of nodes whose outputs are inputs to this node.

    description : str, optional
        Human-readable description of what this node does.

    pre_funcs : list of Callable, optional
        Functions to run before executing the node. Each function receives
        the inputs dictionary. Can be used for validation or inspection.

    post_funcs : list of Callable, optional
        Functions to run after executing the node. Each function receives
        the node output. Can be used for validation or inspection.

    validate_enabled : bool
        Whether pre/post functions are enabled for this node.

    metadata : dict, optional
        Additional metadata/tags for the node.
    """

    name: str
    func: Callable
    dependencies: list[str] = attrs.field(factory=list)
    description: str = ""
    pre_funcs: list[Callable] = attrs.field(factory=list)
    post_funcs: list[Callable] = attrs.field(factory=list)
    validate_enabled: bool = True
    metadata: dict[str, Any] = attrs.field(factory=dict)

    def pprint(self):
        try:
            from rich.pretty import pprint
        except ImportError:
            raise ImportError(
                "rich is required for pretty printing. Install with: pip install rich"
            )

        pprint(self)


@attrs.define
class Pipeline:
    """
    A lightweight DAG-based pipeline engine.

    This class provides an imperative API for building and executing
    computational pipelines. It uses networkx for graph operations and
    supports features like input injection and validation.

    Parameters
    ----------
    validate_globally : bool
        Global flag to enable/disable all pre/post functions.

    Examples
    --------
    Basic usage:

    >>> pipeline = Pipeline()
    >>> pipeline.add_node("a", lambda: 1)
    >>> pipeline.add_node("b", lambda: 2)
    >>> pipeline.add_node("c", lambda a, b: a + b, dependencies=["a", "b"])
    >>> results = pipeline.execute(outputs=["c"])
    >>> print(results["c"])
    3

    Virtual Inputs
    --------------
    Dependencies that don't exist as nodes are automatically treated as virtual
    inputs. These must be provided via the ``inputs`` parameter during execution:

    >>> pipeline = Pipeline()
    >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
    >>> pipeline.get_virtual_inputs()
    ['a']
    >>> results = pipeline.execute(outputs=["b"], inputs={"a": 10})
    >>> results["b"]
    11
    """

    validate_globally: bool = True
    _graph: nx.DiGraph = attrs.field(factory=nx.DiGraph, init=False, repr=False)
    _nodes: dict[str, Node] = attrs.field(factory=dict, init=False, repr=False)
    _virtual_inputs: set[str] = attrs.field(factory=set, init=False, repr=False)
    _cache: dict[str, Any] = attrs.field(factory=dict, init=False, repr=False)

    def add_node(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[list[str]] = None,
        description: str = "",
        pre_funcs: Optional[list[Callable]] = None,
        post_funcs: Optional[list[Callable]] = None,
        validate_enabled: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        outputs: Optional[list[str] | dict[str, str | Callable]] = None,
    ) -> "Pipeline":
        """
        Add a computation node to the pipeline.

        Dependencies that don't exist as nodes are automatically treated as
        virtual inputs that must be provided via ``inputs`` during execution.

        When ``outputs`` is provided, ``func`` is expected to return a dict.
        Each entry in ``outputs`` becomes an independent child node, letting
        downstream nodes depend on individual fields rather than the whole dict.

        Parameters
        ----------
        name : str
            Unique identifier for the node. When ``outputs`` is given, this
            node holds the intermediate dict; by convention, prefix it with an
            underscore (e.g. ``"_stats"``).

        func : Callable
            Function to execute. Its parameters must match dependency names.
            When ``outputs`` is given, must return a dict.

        dependencies : list of str, optional
            Names of nodes or virtual inputs whose outputs feed into this node.

        description : str, optional
            Human-readable description.

        pre_funcs : list of Callable, optional
            Functions to run before node execution (validation, inspection).

        post_funcs : list of Callable, optional
            Functions to run after node execution (validation, inspection).

        validate_enabled : bool
            Enable pre/post functions for this node.

        metadata : dict, optional
            Additional metadata to attach to the node.

        outputs : list of str or dict, optional
            Specifies child nodes to create from the dict returned by ``func``.
            Three forms are accepted:

            * ``list[str]``: each string becomes both the node ID and the dict
              key to extract. ``["x", "y"]`` is equivalent to
              ``{"x": "x", "y": "y"}``.
            * ``dict[str, str]``: maps node ID to dict key.
              ``{"x_node": "x_key"}`` extracts ``d["x_key"]`` into node
              ``"x_node"``.
            * ``dict[str, Callable]``: maps node ID to an extractor callable
              that receives the full dict and returns the node value.
              ``{"x": lambda d: d["x"]}`` for full control.

        Returns
        -------
        Pipeline
            Self for method chaining.

        Raises
        ------
        ValueError
            If node name already exists or if adding creates a cycle.

        Examples
        --------
        Simple node:

        >>> pipeline = Pipeline()
        >>> pipeline.add_node("a", lambda: 1)
        >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        >>> pipeline.execute(outputs=["b"])
        {'b': 2}

        Field extraction — list form (node ID == dict key):

        >>> pipeline = Pipeline()
        >>> pipeline.add_node("_raw", lambda: {"x": 1, "y": 2}, outputs=["x", "y"])
        >>> pipeline.execute(outputs=["x", "y"])
        {'x': 1, 'y': 2}

        Field extraction — dict[str, str] form (node ID → dict key):

        >>> pipeline = Pipeline()
        >>> pipeline.add_node(
        ...     "_raw",
        ...     lambda: {"x_internal": 1},
        ...     outputs={"x": "x_internal"},
        ... )
        >>> pipeline.execute(outputs=["x"])
        {'x': 1}

        Field extraction — dict[str, Callable] form (full control):

        >>> pipeline = Pipeline()
        >>> pipeline.add_node(
        ...     "_raw",
        ...     lambda: {"x": 1, "y": 2},
        ...     outputs={"sum": lambda d: d["x"] + d["y"]},
        ... )
        >>> pipeline.execute(outputs=["sum"])
        {'sum': 3}
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")

        # If this node name was previously a virtual input, it's now a real node
        if name in self._virtual_inputs:
            self._virtual_inputs.remove(name)

        dependencies = dependencies or []

        # Track newly added virtual inputs for potential rollback
        new_virtual_inputs = []

        # Identify virtual inputs: dependencies that don't exist as nodes
        for dep in dependencies:
            if dep not in self._nodes:
                # This is a virtual input
                if dep not in self._virtual_inputs:
                    self._virtual_inputs.add(dep)
                    new_virtual_inputs.append(dep)
                # Add to graph if not already present (for dependency tracking)
                if not self._graph.has_node(dep):
                    self._graph.add_node(dep, node=None)  # None indicates virtual

        # Create node object
        node = Node(
            name=name,
            func=func,
            dependencies=dependencies,
            description=description,
            pre_funcs=pre_funcs or [],
            post_funcs=post_funcs or [],
            validate_enabled=validate_enabled,
            metadata=metadata or {},
        )

        # Add to graph
        self._graph.add_node(name, node=node)
        for dep in dependencies:
            self._graph.add_edge(dep, name)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self._graph):
            # Rollback: remove node and any virtual inputs just added
            self._graph.remove_node(name)
            for dep in new_virtual_inputs:
                # Only remove if this was the only consumer
                if self._graph.has_node(dep) and self._graph.out_degree(dep) == 0:
                    self._graph.remove_node(dep)
                    self._virtual_inputs.remove(dep)
            raise ValueError(f"Adding node '{name}' would create a cycle")

        self._nodes[name] = node

        if outputs is not None:
            if isinstance(outputs, Sequence):
                # ["x", "y"] → {"x": "x", "y": "y"}
                outputs = {x: x for x in outputs}

            # dict: str values become key-extractors, Callable values pass through
            outputs: dict[str, Callable] = {
                node_id: (
                    (lambda d, k=field: d[k]) if isinstance(field, str) else field
                )
                for node_id, field in outputs.items()
            }

            # Wrap each extractor so the engine can call it with **inputs.
            # _execute_node calls func(**{name: dict_value}), but user-supplied
            # extractors expect the dict as a plain positional argument.
            def _make_extractor(src: str, ext: Callable) -> Callable:
                def wrapped(**kwargs: Any) -> Any:
                    return ext(kwargs[src])

                return wrapped

            for output_name, extractor in outputs.items():
                self.add_node(
                    output_name, _make_extractor(name, extractor), dependencies=[name]
                )

        return self

    def remove_node(self, name: str) -> Pipeline:
        """
        Remove a node from the pipeline.

        Parameters
        ----------
        name : str
            Name of the node to remove.

        Returns
        -------
        Pipeline
            Self for method chaining.

        Raises
        ------
        ValueError
            If node doesn't exist or has downstream dependencies.
        """
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")

        # Check if any nodes depend on this one
        successors = list(self._graph.successors(name))
        if successors:
            raise ValueError(
                f"Cannot remove node '{name}': nodes {successors} depend on it"
            )

        # Get dependencies before removing
        node = self._nodes[name]

        # Remove from graph and nodes dict
        self._graph.remove_node(name)
        del self._nodes[name]

        # Clear cache for this node
        if name in self._cache:
            del self._cache[name]

        # Check if any of this node's dependencies were virtual inputs
        # that are now orphaned (no other nodes depend on them)
        for dep in node.dependencies:
            if dep in self._virtual_inputs:
                if not self._graph.has_node(dep) or self._graph.out_degree(dep) == 0:
                    # This virtual input is no longer needed
                    if self._graph.has_node(dep):
                        self._graph.remove_node(dep)
                    self._virtual_inputs.remove(dep)

        return self

    def extract_subgraph(self, outputs: list[str]) -> Pipeline:
        """
        Extract a subgraph containing only nodes needed for specified outputs.

        Virtual inputs required for the outputs are preserved in the subgraph.

        Parameters
        ----------
        outputs : list of str
            Names of output nodes to include.

        Returns
        -------
        Pipeline
            A new pipeline containing only necessary nodes and virtual inputs.

        Raises
        ------
        ValueError
            If any output node doesn't exist.
        """
        for output in outputs:
            if output not in self._nodes:
                raise ValueError(f"Output node '{output}' not found")

        # Find all ancestors of the output nodes
        required_nodes = set()
        required_virtual_inputs = set()

        for output in outputs:
            required_nodes.add(output)
            ancestors = nx.ancestors(self._graph, output)

            for ancestor in ancestors:
                if ancestor in self._nodes:
                    required_nodes.add(ancestor)
                elif ancestor in self._virtual_inputs:
                    required_virtual_inputs.add(ancestor)

        # Create new pipeline
        new_pipeline = Pipeline(validate_globally=self.validate_globally)

        # First, add virtual inputs to the new pipeline
        for vi in required_virtual_inputs:
            new_pipeline._virtual_inputs.add(vi)
            new_pipeline._graph.add_node(vi, node=None)

        # Add nodes in topological order
        for node_name in nx.topological_sort(self._graph):
            if node_name in required_nodes:
                node = self._nodes[node_name]
                new_pipeline.add_node(
                    name=node.name,
                    func=node.func,
                    dependencies=node.dependencies,
                    description=node.description,
                    pre_funcs=node.pre_funcs,
                    post_funcs=node.post_funcs,
                    validate_enabled=node.validate_enabled,
                    metadata=node.metadata,
                )

        return new_pipeline

    def execute(
        self,
        outputs: Optional[list[str]] = None,
        inputs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute the pipeline and return results.

        Parameters
        ----------
        outputs : list of str, optional
            Names of nodes whose values should be returned. These can be any
            nodes in the pipeline, including intermediate ones. All required
            ancestor nodes will be computed automatically.
            If None, computes all leaf nodes.

        inputs : dict, optional
            Dictionary mapping node names or virtual inputs to data values.

            - For virtual inputs: provides the required input values.
            - For regular nodes: the node will not be executed; the provided
              value is used instead, effectively bypassing its computation.

        Returns
        -------
        dict
            Dictionary mapping requested node names to their computed values.

        Raises
        ------
        ValueError
            If output nodes don't exist, if required virtual inputs are missing,
            or if outputs are not reachable from provided inputs.
        """
        inputs = inputs or {}

        # Determine which nodes to compute
        if outputs is None:
            # Find leaf nodes (no successors), excluding virtual inputs
            outputs = [n for n in self._nodes if self._graph.out_degree(n) == 0]

        # Validate outputs exist
        for output in outputs:
            if output not in self._nodes:
                raise ValueError(f"Output node '{output}' not found")

        # Separate inputs into node bypasses and virtual input values
        node_bypasses = {}
        virtual_input_values = {}

        for key, value in inputs.items():
            if key in self._nodes:
                node_bypasses[key] = value
            elif key in self._virtual_inputs:
                virtual_input_values[key] = value
            else:
                raise ValueError(
                    f"Input key '{key}' is neither a node nor a virtual input"
                )

        # Determine required virtual inputs for requested outputs
        required_virtual_inputs = self._get_required_virtual_inputs(
            outputs, node_bypasses
        )

        # Validate all required virtual inputs are provided
        missing_inputs = required_virtual_inputs - set(virtual_input_values.keys())
        if missing_inputs:
            raise ValueError(
                f"Missing required virtual inputs: {sorted(missing_inputs)}. "
                f"These must be provided in inputs."
            )

        # Validate connectivity: outputs must be reachable from virtual inputs
        # + bypassed nodes
        self._validate_connectivity(outputs, virtual_input_values, node_bypasses)

        # Clear cache
        self._cache.clear()

        # Add input data to cache (both node bypasses and virtual inputs)
        self._cache.update(node_bypasses)
        self._cache.update(virtual_input_values)

        # Determine execution order
        required_nodes = set()
        for output in outputs:
            required_nodes.add(output)
            # Only add ancestors that aren't bypassed or virtual
            for ancestor in nx.ancestors(self._graph, output):
                if ancestor not in inputs and ancestor not in self._virtual_inputs:
                    required_nodes.add(ancestor)

        # Get topological order (exclude virtual inputs and bypassed nodes)
        execution_order = [
            n
            for n in nx.topological_sort(self._graph)
            if n in required_nodes and n not in inputs and n not in self._virtual_inputs
        ]

        # Execute nodes
        for node_name in execution_order:
            self._execute_node(node_name)

        # Return requested outputs
        return {name: self._cache[name] for name in outputs}

    def _execute_node(self, node_name: str) -> Any:
        """
        Execute a single node.

        Parameters
        ----------
        node_name : str
            Name of the node to execute.

        Returns
        -------
        Any
            The computed result.
        """
        if node_name in self._cache:
            return self._cache[node_name]

        node = self._nodes[node_name]

        # Gather inputs
        inputs = {}
        for dep in node.dependencies:
            # Recursively execute dependencies if not cached
            if dep not in self._cache:
                self._execute_node(dep)
            inputs[dep] = self._cache[dep]

        # Run pre-funcs
        if self.validate_globally and node.validate_enabled:
            for func in node.pre_funcs:
                func(inputs)

        # Execute node function
        result = node.func(**inputs)

        # Run post-funcs
        if self.validate_globally and node.validate_enabled:
            for func in node.post_funcs:
                func(result)

        # Cache result
        self._cache[node_name] = result

        return result

    def _get_required_virtual_inputs(
        self, outputs: list[str], node_bypasses: dict[str, Any]
    ) -> set[str]:
        """Determine which virtual inputs are required for given outputs.

        Parameters
        ----------
        outputs : list of str
            Output nodes to compute.

        node_bypasses : dict
            Nodes being bypassed (don't need their ancestors).

        Returns
        -------
        set of str
            Virtual input names required for computing outputs.
        """
        required = set()

        for output in outputs:
            # Find all ancestors of this output
            ancestors = nx.ancestors(self._graph, output)

            for ancestor in ancestors:
                # If ancestor is a virtual input and not bypassed, it's required
                if ancestor in self._virtual_inputs and ancestor not in node_bypasses:
                    # Check if there's a path from this virtual input to output
                    # that doesn't go through bypassed nodes
                    if self._is_reachable_without_bypass(
                        ancestor, output, node_bypasses
                    ):
                        required.add(ancestor)

        return required

    def _is_reachable_without_bypass(
        self, source: str, target: str, node_bypasses: dict[str, Any]
    ) -> bool:
        """
        Check if target is reachable from source without going through bypasses.

        Parameters
        ----------
        source : str
            Source node name.

        target : str
            Target node name.

        node_bypasses : dict
            Bypassed nodes to exclude from path.

        Returns
        -------
        bool
            True if target is reachable from source.
        """
        # Use BFS to find if there's a path
        visited = {source}
        queue = [source]

        while queue:
            current = queue.pop(0)

            if current == target:
                return True

            for successor in self._graph.successors(current):
                # Skip bypassed nodes (but not if it's the target)
                if successor in node_bypasses and successor != target:
                    continue

                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return False

    def _validate_connectivity(
        self,
        outputs: list[str],
        virtual_input_values: dict[str, Any],
        node_bypasses: dict[str, Any],
    ) -> None:
        """
        Validate that outputs are reachable from provided inputs.

        This ensures the pipeline execution is well-formed: all outputs must be
        computable from the combination of:

        - Virtual inputs with provided values
        - Bypassed nodes with provided values
        - Regular nodes that will be executed

        Parameters
        ----------
        outputs : list of str
            Output nodes to compute.

        virtual_input_values : dict
            Virtual inputs with provided values.

        node_bypasses : dict
            Nodes being bypassed with provided values.

        Raises
        ------
        ValueError
            If any output is not reachable from provided inputs.
        """
        # For each output, verify there's a path from some root
        # Roots are: virtual inputs (with values) + bypassed nodes
        # + parameter-less nodes
        roots = set(virtual_input_values.keys()) | set(node_bypasses.keys())

        # Add nodes with no dependencies (parameter-less functions)
        for node_name in self._nodes:
            if self._graph.in_degree(node_name) == 0:
                roots.add(node_name)

        # Check each output
        for output in outputs:
            # Find all ancestors
            ancestors = nx.ancestors(self._graph, output)
            ancestors.add(output)

            # Check if this subgraph is connected to any root
            has_root = False
            for node in ancestors:
                if node in roots:
                    has_root = True
                    break
                # Check if node has no inputs (is itself a root)
                if node in self._nodes and self._graph.in_degree(node) == 0:
                    has_root = True
                    break

            if not has_root:
                # Find which virtual inputs are in the ancestry
                virtual_ancestors = ancestors & self._virtual_inputs
                missing = sorted(virtual_ancestors - set(virtual_input_values.keys()))
                raise ValueError(
                    f"Output '{output}' is not reachable from provided inputs. "
                    f"The following virtual inputs in its dependency chain "
                    f"have no values: {missing}"
                )

    def get_node(self, name: str) -> Node:
        """
        Get a node by name.

        Parameters
        ----------
        name : str
            Name of the node.

        Returns
        -------
        Node
            The node object.

        Raises
        ------
        ValueError
            If node doesn't exist.
        """
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        return self._nodes[name]

    def list_nodes(self) -> list[str]:
        """
        List all node names in topological order.

        Returns
        -------
        list of str
            Node names in topological order.
        """
        return list(nx.topological_sort(self._graph))

    def get_nodes_by_metadata(self, **kwargs: Any) -> list[str]:
        """
        Return names of nodes whose metadata matches all given key-value pairs.

        Virtual inputs are excluded. Results are returned in topological order.

        Parameters
        ----------
        **kwargs
            Key-value pairs that must all be present in a node's metadata.

        Returns
        -------
        list of str
            Node names (in topological order) whose metadata contains all
            specified key-value pairs.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> pipeline.add_node("a", lambda: 1, metadata={"final": True, "kind": "data"})
        >>> pipeline.add_node("b", lambda: 2, metadata={"final": True, "kind": "debug"})
        >>> pipeline.get_nodes_by_metadata(final=True, kind="data")
        ['a']
        >>> pipeline.get_nodes_by_metadata(final=True)
        ['a', 'b']
        """
        result = []
        for name in nx.topological_sort(self._graph):
            if name in self._virtual_inputs:
                continue
            node = self._nodes[name]
            if all(node.metadata.get(k) == v for k, v in kwargs.items()):
                result.append(name)
        return result

    def get_virtual_inputs(self) -> list[str]:
        """
        Get all virtual input names in the pipeline.

        Virtual inputs are dependencies that don't exist as nodes and must be
        provided via ``inputs`` during execution.

        Returns
        -------
        list of str
            Virtual input names in sorted order.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        >>> pipeline.get_virtual_inputs()
        ['a']
        """
        return sorted(self._virtual_inputs)

    def get_required_inputs(
        self,
        outputs: Optional[list[str]] = None,
        inputs: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """
        Get virtual inputs required for computing specific outputs.

        This helps users determine what data they need to provide in ``inputs``
        to execute specific outputs.

        Parameters
        ----------
        outputs : list of str, optional
            Output nodes to compute. If None, uses all leaf nodes.
        inputs : dict, optional
            Nodes/inputs being provided. These reduce required inputs.

        Returns
        -------
        list of str
            Required virtual input names in sorted order.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        >>> pipeline.add_node("c", lambda b: b * 2, dependencies=["b"])
        >>> pipeline.get_required_inputs(outputs=["c"])
        ['a']
        >>> pipeline.get_required_inputs(outputs=["c"], inputs={"b": 5})
        []
        """
        inputs = inputs or {}

        # Determine outputs
        if outputs is None:
            outputs = [n for n in self._nodes if self._graph.out_degree(n) == 0]

        # Validate outputs exist
        for output in outputs:
            if output not in self._nodes:
                raise ValueError(f"Output node '{output}' not found")

        # Separate node bypasses from virtual input values
        node_bypasses = {k: v for k, v in inputs.items() if k in self._nodes}

        # Get required virtual inputs
        required = self._get_required_virtual_inputs(outputs, node_bypasses)

        return sorted(required)

    def is_virtual_input(self, name: str) -> bool:
        """
        Check if a name corresponds to a virtual input.

        Parameters
        ----------
        name : str
            Name to check.

        Returns
        -------
        bool
            True if name is a virtual input.
        """
        return name in self._virtual_inputs

    def clear_cache(self) -> None:
        """Clear the execution cache."""
        self._cache.clear()

    def set_global_validation(self, enabled: bool) -> None:
        """
        Enable or disable validation globally.

        Parameters
        ----------
        enabled : bool
            Whether to enable validation.
        """
        self.validate_globally = enabled

    def _to_dot(
        self, highlight_nodes: Optional[list[str]] = None, legend: bool = False
    ):
        """
        Build a pydot graph representation of the pipeline.

        Parameters
        ----------
        highlight_nodes : list of str, optional
            Node names to highlight in the visualization.

        legend : bool
            If True, add a legend explaining node styles.

        Returns
        -------
        pydot.Dot
            The constructed graph object.

        Raises
        ------
        ImportError
            If pydot is not installed.
        """
        try:
            import pydot
        except ImportError:
            raise ImportError(
                "pydot is required for visualization. Install with: pip install pydot"
            )

        highlight_nodes = set(highlight_nodes or [])

        dot_graph = pydot.Dot(
            graph_type="digraph",
            rankdir="TB",
            fontname="Helvetica",
            fontsize="10",
        )

        # Set default font
        dot_graph.set_node_defaults(**_DOT_STYLES["default_node"])
        dot_graph.set_edge_defaults(**_DOT_STYLES["default_edge"])

        for node_name in self.list_nodes():
            # Check if this is a virtual input or a real node
            if node_name in self._virtual_inputs:
                # Virtual input styling
                style_attrs = {
                    **_DOT_STYLES["node_virtual_input"],
                    "label": f'< <FONT FACE="Courier" POINT-SIZE="12"><B>{node_name}'
                    "</B></FONT> >",
                }

            else:
                # Regular node
                node = self.get_node(node_name)

                # Build label with HTML-like syntax for mixed fonts
                label_parts = [
                    # Titile with fixed-width bold font
                    f'<FONT FACE="Courier" POINT-SIZE="12"><B>{node_name}'
                    "</B></FONT><BR/>"
                ]

                if node.description:
                    # Wrap long descriptions
                    words = node.description.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(" ".join(current_line + [word])) > 30:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                        else:
                            current_line.append(word)
                    if current_line:
                        lines.append(" ".join(current_line))
                    # Description in regular Helvetica
                    for line in lines:
                        label_parts.append(line)

                # Add metadata tags in italic Helvetica
                if node.metadata:
                    tags = ", ".join(f"{k}: {v}" for k, v in node.metadata.items())
                    label_parts.append(f"<I>{{{tags}}}</I>")

                # Combine parts with line breaks
                label = "< " + "<BR/>".join(label_parts) + " >"

                # Styling
                style_attrs = {**_DOT_STYLES["node_computation"], "label": label}

            if node_name in highlight_nodes:
                style_attrs.update(_DOT_STYLES["node_highlight"])

            dot_node = pydot.Node(node_name, **style_attrs)
            dot_graph.add_node(dot_node)

        for edge in self._graph.edges():
            dot_graph.add_edge(pydot.Edge(edge[0], edge[1]))

        if legend:
            legend_graph = pydot.Cluster(
                "legend", label="< <B>Legend</B> >", **_DOT_STYLES["legend_box"]
            )
            legend_graph.add_node(
                pydot.Node(
                    "legend_virtual",
                    label="Virtual\ninput",
                    **_DOT_STYLES["node_virtual_input"],
                )
            )
            legend_graph.add_node(
                pydot.Node(
                    "legend_node",
                    label="Computation\nnode",
                    **_DOT_STYLES["node_computation"],
                )
            )

            dot_graph.add_subgraph(legend_graph)  # type: ignore[arg-type]

        return dot_graph

    def visualize(
        self,
        highlight_nodes: Optional[list[str]] = None,
        legend: bool = False,
    ):
        """
        Generate and display pipeline visualization as SVG in Jupyter notebooks.

        This function creates an SVG visualization using the Graphviz dot backend
        and displays it inline in Jupyter notebooks using IPython.display.

        Parameters
        ----------
        highlight_nodes : list of str, optional
            Node names to highlight in the visualization.

        legend : bool
            If True, add a legend explaining node styles.

        Returns
        -------
        IPython.display.SVG
            SVG object that will display inline in Jupyter notebooks.

        Raises
        ------
        ImportError
            If pydot or IPython is not installed.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> pipeline.add_node("a", lambda: 1)
        >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        >>> pipeline.visualize()  # doctest: +SKIP
        """
        try:
            from IPython.display import SVG
        except ImportError:
            raise ImportError(
                "IPython is required for notebook display. "
                "Install with: pip install ipython"
            )

        dot_graph = self._to_dot(highlight_nodes, legend=legend)
        svg_data = dot_graph.create_svg()

        return SVG(svg_data)

    def write_dot(
        self,
        filename: str,
        highlight_nodes: Optional[list[str]] = None,
        legend: bool = False,
    ) -> None:
        """
        Export pipeline to Graphviz DOT format with advanced styling.

        Parameters
        ----------
        filename : str
            Output filename.

        highlight_nodes : list of str, optional
            Node names to highlight in the visualization.

        legend : bool
            If True, add a legend explaining node styles.

        Raises
        ------
        ImportError
            If pydot is not installed.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> pipeline.add_node("a", lambda: 1)
        >>> pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])
        >>> pipeline.export_to_dot("my_pipeline.dot")
        """
        dot_graph = self._to_dot(highlight_nodes, legend=legend)
        dot_graph.write(filename)

    def write_png(
        self,
        filename: str,
        highlight_nodes: Optional[list[str]] = None,
        legend: bool = False,
    ) -> None:
        """Export pipeline visualization to a PNG file.

        Parameters
        ----------
        filename : str
            Output PNG filename.

        highlight_nodes : list of str, optional
            Node names to highlight in the visualization.

        legend : bool
            If True, add a legend explaining node styles.

        Raises
        ------
        ImportError
            If pydot is not installed.
        """
        dot_graph = self._to_dot(highlight_nodes, legend=legend)
        dot_graph.write_png(filename)

    def write_svg(
        self,
        filename: str,
        highlight_nodes: Optional[list[str]] = None,
        legend: bool = False,
    ) -> None:
        """Export pipeline visualization to an SVG file.

        Parameters
        ----------
        filename : str
            Output SVG filename.

        highlight_nodes : list of str, optional
            Node names to highlight in the visualization.

        legend : bool
            If True, add a legend explaining node styles.

        Raises
        ------
        ImportError
            If pydot is not installed.
        """
        dot_graph = self._to_dot(highlight_nodes, legend=legend)
        dot_graph.write_svg(filename)

    def _repr_svg_(self) -> str:
        """Return SVG representation for Jupyter notebook auto-display.

        Returns
        -------
        str
            SVG markup string.
        """
        try:
            dot_graph = self._to_dot()
            return dot_graph.create_svg().decode("utf-8")
        except Exception:
            return ""

    def print_summary(self) -> None:
        """
        Print a text summary of the pipeline structure.
        """
        print("Pipeline Summary")
        print("=" * 50)
        print(f"Nodes: {len(self._nodes)}")
        print(f"Validation: {'Enabled' if self.validate_globally else 'Disabled'}")
        print()
        print("Execution Order:")

        for i, node_name in enumerate(self.list_nodes(), 1):
            node = self.get_node(node_name)
            desc = f" - {node.description}" if node.description else ""
            print(f"{i}. {node_name}{desc}")

            if node.dependencies:
                print(f"   Dependencies: {', '.join(node.dependencies)}")

            if node.metadata:
                tags = ", ".join(f"{k}={v}" for k, v in node.metadata.items())
                print(f"   Metadata: {tags}")

            if node.pre_funcs or node.post_funcs:
                func_count = len(node.pre_funcs) + len(node.post_funcs)
                print(f"   Pre/post funcs: {func_count}")
