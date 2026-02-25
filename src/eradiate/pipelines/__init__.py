"""
This module contains the infrastructure, definitions, and logic for all
post-processing pipeline operations. The pipeline is modelled as a directed
acyclic graph (DAG) built with a lightweight custom engine.

For clarity, the implementation is split as follows:

* :mod:`eradiate.pipelines.engine` provides the fundamental logic powering each
* :mod:`eradiate.pipelines.logic` provides the fundamental logic powering each
  step of post-processing;
* :mod:`eradiate.pipelines.definitions` contains the imperative pipeline builder
  (:func:`~eradiate.pipelines.definitions.build_pipeline`);
* :mod:`eradiate.pipelines._config` (private) provides the
  :func:`~eradiate.pipelines.config` utility for generating pipeline
  configuration dictionaries.
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
