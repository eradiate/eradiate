.. _sec-developer_guides-lazy_loading:

Lazy module imports
===================

Python modules may become large enough for import-time code execution to have a
perceptible overhead. This applies to Eradiate, whose codebase and dependency
stack are big enough for imports on a high-performance desktop CPU at the time
of writing (AMD Ryzen 9 5900X) to take more than a second.

Code optimization may not even completely solve the issue: unavoidable
dependencies may have significant overhead. This used to apply, for instance,
`to xarray <https://github.com/pydata/xarray/issues/6726>`_, ubiquitous in our
codebase.

The solution the community seems to settle with is the lazy import of modules:
module import-time code is executed upon module access, rather than upon import.
The `Scientific Python <https://scientific-python.org/>`_ community came up with
an `Ecosystem Coordination proposal <https://scientific-python.org/specs/spec-0001/>`_
aiming to address this very issue. We use a
`publicly available implementation <https://github.com/scientific-python/lazy_loader>`_
to both implement component lazy loading and populate our public API.

Using ``lazy_loader``
---------------------

Usage is documented on the
`GitHub repository <https://github.com/scientific-python/lazy_loader/>`_.
Be sure to read the **Type checkers** section carefully as it describes how we
use stub files.

In addition, checking that all lazy imports can be resolved is recommended. The
way to do so is to use the ``EAGER_IMPORT`` environment variable, *e.g.*

.. code::

   EAGER_IMPORT=1 python -c "import eradiate"

This will force eager import of lazy module definitions. Missing modules will
result in a :class:`ModuleNotFoundError` being raised.
