"""
This module contains the infrastructure, definitions, and logic for all
post-processing pipeline operations. The implementation complies with the
`Hamilton <https://hamilton.dagworks.io/>`__ dataflow framework's specifications:
the post-processing pipeline is modelled as a directed acyclic graph (DAG), and
all post-processing operations are a node in that DAG.

For clarity, the implementation is split as follows:

* :mod:`eradiate.pipelines.logic` provides the fundamental logic powering each
  step of post-processing;
* :mod:`eradiate.pipelines.definitions` contains the pipeline definition and is
  intended to be consumed by the Hamilton driver constructor for pipeline
  initialization;
* :mod:`eradiate.pipelines.core` provides convenience entry points to facilitate
  post-processing pipeline initialization.

See Also
--------

* :class:`hamilton.driver.Driver`
* `Hamilton's documentation on tags \
  <https://hamilton.dagworks.io/en/latest/reference/decorators/tag/#tag>`__
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
