.. _sec-developer_guides-lazy_loading:

Lazy module imports
===================

Python modules may become large enough for import-time code execution to have a
perceptible overhead. This applies to Eradiate, whose codebase and dependency
stack are big enough for imports on a high-performance desktop CPU at the time
of writing (AMD Ryzen 9 5900X) to take more than a second.

Code optimisation may not even completely solve the issue: unavoidable
dependencies may have significant overhead. This applies, in particular,
`to xarray <https://github.com/pydata/xarray/issues/6726>`_.

The solution the community seems to settle with is the lazy import of modules:
module import-time code is executed upon module access, rather than upon import.
The `Scientific Python <https://scientific-python.org/>`_ community came up with
an `Ecosystem Coordination proposal <https://scientific-python.org/specs/spec-0001/>`_
aiming to address this very issue. An implementation is
`available publicly <https://github.com/scientific-python/lazy_loader>`_ and
used in Eradiate, available as the :mod:`eradiate.util.lazy_loader` module.

.. admonition:: Why vendor rather than depend?
   :class: note

   As of the time of writing, the ``lazy_loader`` package is unstable; however,
   ``lazy_loader`` ambitions to be a standard for lazy imports (at least from
   the understanding of the Eradiate authors) and should therefore become a
   dependency of many packages.

   Lack of stability would therefore advocate for pinning this dependency; but
   (potentially) wide adoption advises against it to avoid clashes. Also, this
   package is not distributed on conda-forge, which is problematic for Eradiate.

   Therefore, it was decided that the simplest way to implement this feature
   without risking dependency clashes was to vendor this code. We will
   transition to a dependency-based pattern when ``lazy_loader`` will be stable
   and packaged on conda-forge.

Using ``lazy_loader``
---------------------

Usage is documented in the
`SPEC <https://scientific-python.org/specs/spec-0001/>`_. Docstrings also
provide useful usage information. Be sure to read the **TYPE CHECKERS** section
carefully as it describes how we use stub files.

In addition, checking that all lazy imports can be resolved is recommended. The
way to do so is to use the ``EAGER_IMPORT`` environment variable, *e.g.*

.. code::

   EAGER_IMPORT=1 python -c "import eradiate"

This will force eager import of lazy module definitions. Missing modules will
result in a :class:`ModuleNotFoundError` being raised.
