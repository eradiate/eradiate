####################
Eradiate Test Report
####################

This report provides an overview of all tests executed to assert the function of
Eradiate. 

Eradiate tests are split in two groups. **Unit tests** are short and simple test cases,
which test the correct function of small units of code, such as individual class
methods or functions. These tests usually instantiate one class of Eradiate and 
call its methods. **System tests** on the other hand are more complex and test
the whole system from its inputs to its outputs. These tests usually start with 
the kind of input a user would provide to the system, such as an XML file, and run
their tests on the final result of the computation.

The following summary table gives a short overview of the number of tests and their
results. Through the navigation bar and by clicking on the result category in the 
table, you can access additional detailed information
on the number of passed, failed and skipped tests.

The section **Test specification** contains the list of system tests, shown individually
with a detailed description of their setup and execution.

The section **Benchmarks** contains the results of different benchmarking procedures.

.. only:: not latex

.. include:: generated/summary.rst

.. toctree::
    :hidden:
    :maxdepth: 1

    ../generated/passed

    ../generated/failed

    ../generated/skipped

.. toctree::
    :hidden:
    :maxdepth: 2

    ../generated/testspec
    benchmarks