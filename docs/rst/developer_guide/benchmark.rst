Benchmarking
============

Run Benchmarks
--------------
Eradiate's benchmarking suite is based on the ASV package which enables
tracking a project's performance over time.
To run the benchmark suite:

.. code:: bash

    conda activate <env name>
    cd benchmarks
    python -m eradiate.test_tools.benchmark.benchmark

By default, it will run the benchmark using the currently active environment.
For dev environments, this means that you need an operational version
of eradiate. Note that environments that use production versions prior
to v0.28.xx cannot be benchmarked as they do no include changes required
for the benchmark.

If you want to archive results to a separate **asv database**, add the
``--archive-dir=<dir>`` argument. The path should point to the root
of the database, where the asv.conf.json exists.

ASV also offers the possibility to run benchmarks on previous commits
and branches. It will clone and build the project in a new environment
and run the benchmarks on those. To use this, specify the ``--git-ref=<range>``
as a range, e.g. ``main^!`` for the last commit of the ``main`` branch:

.. code:: bash

    pyhton -m eradiate.test_tools.benchmark --git-ref=main^!

The benchmark results are stored in the ``results`` folder. Those can be
publish to an html page and viewed by running:

.. code:: bash

    # publish results to html page
    asv publish
    # start a server to view results
    asv preview

ASV commands are also available, for custom use of ASV, please refer
to `the package documentation <https://asv.readthedocs.io/en/v0.6.1/>`_.

Write Benchmarks
----------------

Benchmarks are in the ``benchmarks`` folder. They follow the
`ASV syntax <https://asv.readthedocs.io/en/v0.6.1/writing_benchmarks.html>`_.
The test cases that are benchmarks are often also used for regression test.
For this reason, they are stored in ``src/eradiate/test_tools/test_cases/``.
