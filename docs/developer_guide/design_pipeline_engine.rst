.. _sec-developer_guide-design_pipeline_engine:

Design note: Pipeline engine
============================

Motivation
----------

Eradiate uses computational pipelines for postprocessing simulation results.
These pipelines involve chained operations (CKD quadrature aggregation, spectral
response function application, BRDF computation, etc.) organized as directed
acyclic graphs (DAG).

Previously, Eradiate used `Hamilton <https://github.com/dagworks-inc/hamilton>`__
for pipeline management. While Hamilton is feature-rich, several friction points
motivated exploring an alternative:

1. **Steep learning curve**: Heavy reliance on decorators (``@config.when``,
   ``@resolve``, ``@parameterize``, ``@extract_fields``) requires significant
   upfront investment.
2. **Implicit dependencies**: Function parameter names implicitly define the DAG
   structure, making the graph hard to reason about.
3. **Limited dynamic construction**: Building pipelines conditionally requires
   working around decorator semantics.
4. **Debugging difficulty**: Stacked decorators obscure the execution flow.

In addition, Hamilton does not receive as much care as we would hope, which
occasionally resulted in critical issues remaining unaddressed --- and us having
to contribute fixing a codebase we do not know well to do so efficiently.

After looking for alternatives, it turned out that no well-maintained,
off-the-shelf library existed that implements the feature set we needed.
We eventually decided to write our own pipeline engine, effectively going back
to a home-grown solution, benefitting from our experience with Hamilton.

Design goals
------------

- **Simplicity**: Clear, explicit API with minimal indirection.
- **Flexibility**: Build pipelines programmatically.
- **Debuggability**: Straightforward call stack, no decorator layers.
- **Completeness**: Pre/post hooks, input injection, subgraph extraction,
  visualization.

Architecture
------------

The engine is built on `networkx <https://networkx.org/>`_ and consists of two
core types:

:class:`~eradiate.pipelines.engine.Pipeline` : DAG manager and executor.
    This class manages a ``networkx.DiGraph`` and a node registry. Its internal
    state is:

    - **_graph**: A :class:`networkx.DiGraph` instance that holds the DAG
      structure (includes both computation nodes and virtual input
      placeholders).
    - **_nodes**: A ``dict[str, Node]`` that maps names to ``Node`` objects.
    - **_virtual_inputs**: A ``set[str]`` that tracks dependencies with no
      backing node.
    - **_cache**: A ``dict[str, Any]`` which holds per-execution result cache.
    - **validate**: Global toggle for pre/post functions.


:class:`~eradiate.pipelines.engine.Node` : Single computation step.
    An `attrs <https://www.attrs.org/>`_-decorated class representing a
    computation step. Its internal state is:

    - **func**: The callable to execute. Its parameter names must match the
      names of its dependencies.
    - **dependencies**: Explicit list of upstream node or virtual input names.
    - **pre_funcs / post_funcs**: Hook lists for validation, logging, or
      inspection. Users supply plain callables; no built-in validator factories
      are provided. Controlled by the per-node ``validate`` flag and the
      pipeline-level ``validate`` flag.
    - **metadata**: Arbitrary key-value tags (used in visualization and for
      user-defined queries).

Key design decisions
--------------------

Imperative API over decorators
    The engine uses method calls
    (:meth:`pipeline.add_node(...) <.Pipeline.add_node>`) rather than decorators
    to define nodes. This makes conditional logic, dynamic construction, and
    debugging straightforward::

        # Plain Python: no framework DSL
        if mode == "ckd":
            pipeline.add_node("aggregate", aggregate_ckd, dependencies=["raw"])
        else:
            pipeline.add_node("aggregate", lambda raw: raw, dependencies=["raw"])

    Compared to Hamilton's equivalent::

        @config.when(mode="ckd")
        def aggregate(raw): ...

        @config.when(mode="mono")
        def aggregate(raw): ...

Explicit dependencies
    Dependencies are declared as a list of names rather than inferred from
    function parameter names. This decouples the function signature from the
    graph structure and makes the DAG explicit.

Virtual inputs
    Dependencies referencing non-existent nodes are automatically classified as
    **virtual inputs** and tracked in a separate set. This emerged as a natural
    generalization: rather than requiring all source data to be wrapped in
    no-argument nodes, external data can be injected at execution time via the
    ``inputs`` parameter.

    Virtual inputs are represented in the graph as placeholder nodes (stored
    with ``node=None`` in graph data) so that networkx algorithms (topological
    sort, ancestor queries) work uniformly.

    A virtual input can later be "promoted" to a real computation node by
    calling :meth:`~.Pipeline.add_node` with the same name.

Unified ``inputs`` parameter
    The ``inputs`` parameter dict to :meth:`~.Pipeline.execute` serves two
    purposes:

    - **virtual input values**: provide data for dependencies without backing
      nodes;
    - **node bypasses**: provide pre-computed values for existing nodes,
      skipping their execution and excluding their upstream dependencies from
      the execution plan.

    The :meth:`~.Pipeline.execute` method distinguishes between the two by
    checking whether the key exists in ``_nodes`` or ``_virtual_inputs``.

Generalized pre/post hooks
    The pre/post hooks are implemented in a very flexible manner, allowing to
    perform data validation during execution, log, or transform data (the latter
    being discouraged). **Pre-functions** operate on the inputs to the node's
    function, while  **post-functions** operator on the node output.

    Global and per-node toggle (both ``validate``) control whether hooks run. No
    built-in validator factory functions are provided; users supply plain
    callables directly.

Visualization integrated into pipeline
    Export methods (``write_dot``, ``write_png``, ``write_svg``, ``visualize``)
    live directly on :class:`~eradiate.pipelines.engine.Pipeline``. This keeps
    the API cohesive and supports Jupyter auto-display via ``_repr_svg_()``.
    Visualization uses `pydot <https://github.com/pydot/pydot>`_ (optional
    dependency).

Execution Algorithm
-------------------

1. **Determine outputs**: Use requested outputs, or default to all leaf nodes.
2. **Classify inputs**: Split ``inputs`` into node bypasses vs. virtual input
   values; reject unknown keys.
3. **Validate virtual inputs**: Determine which virtual inputs are required
   (considering bypasses that may eliminate upstream paths) and ensure all are
   provided.
4. **Validate connectivity**: Confirm all outputs are reachable from the
   combination of roots (parameter-less nodes), virtual inputs, and bypasses.
5. **Clear cache**: Each ``execute()`` starts fresh.
6. **Populate cache**: Insert bypass values and virtual input values.
7. **Compute execution order**: Topological sort, filtering to only nodes in
   the dependency chain of requested outputs (excluding bypassed and virtual
   nodes).
8. **Execute nodes**: For each node in order:

   - Gather inputs from cache.
   - Run pre-functions (if validation enabled).
   - Call ``node.func(**inputs)``.
   - Run post-functions (if validation enabled).
   - Cache result.

9. **Return results**: Extract requested outputs from cache.

Recursive dependency resolution in ``_execute_node`` handles edge cases where
topological order alone isn't sufficient (*e.g.* subgraph boundaries).

Comparison with Hamilton
------------------------

.. list-table::
    :header-rows: 1
    :widths: 25 35 40

    * - Aspect
      - Hamilton
      - Pipeline Engine
    * - API style
      - Decorator-based
      - Imperative method calls
    * - Dependencies
      - Implicit (parameter names)
      - Explicit list
    * - Conditional nodes
      - ``@config.when()``
      - Python ``if``/``else``
    * - Dynamic construction
      - Difficult
      - Natural
    * - Pre/post hooks
      - Separate add-on
      - Built-in; user supplies callables
    * - Subgraph extraction
      - Not built-in
      - :meth:`~.Pipeline.extract_subgraph()`
    * - Input injection
      - Override inputs dict
      - Unified ``inputs`` parameter
    * - Debugging
      - Complex (decorator stack)
      - Standard Python call stack

Dependencies
------------

- **networkx** (required): Topological sort, cycle detection, ancestor queries.
  All graph operations are O(V + E).
- **pydot** (optional): Graphviz DOT generation for visualization.
- **IPython** (optional): Jupyter notebook inline display.

Design evolution
----------------

The implementation went through several iterations:

1. **Initial implementation**: Core Pipeline/Node with ``bypass_data``,
   ``add_interceptor()``, ``pre_validators``/``post_validators``, separate
   visualization module.
2. **Virtual inputs**: Added automatic virtual input detection, introspection
   methods, refactored to attrs.
3. **Visualization consolidation**: Integrated visualization into Pipeline,
   added legend support, Jupyter auto-display.
4. **API simplification**: Renamed ``bypass_data`` to ``inputs``, removed
   ``add_interceptor()``, generalized validators to
   ``pre_funcs``/``post_funcs``, removed built-in validator factories (likely
   reintroduced later, but at the time, no validator was used anywhere in any
   post-processing pipeline).
