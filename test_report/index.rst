####################
Eradiate Test Report
####################

This report provides an overview of all tests executed to assert the function of
Eradiate. 

Eradiate tests are split in two groups:

* **Unit tests** are short and simple test cases, which test the correct function
  of small units of code, such as individual class methods or functions. These
  tests usually instantiate one class of Eradiate and call its methods.
* **System tests** on the other hand are more complex and test the whole system
  from its inputs to its outputs. These tests usually start with the kind of
  input a user would provide to the system, such as an XML file, and run
  their tests on the final result of the computation.

The :ref:`sec-summary` section gives a short overview of the number of tests and their
results. From there, you can access additional detailed information on the
actual passed, failed and skipped tests.

The :ref:`sec-testspec` section contains the list of system tests, shown individually
with a detailed description of their setup and execution.

The :ref:`sec-benchmarks` section contains the results of different benchmarking procedures.

.. include:: generated/summary.rst

.. include:: generated/passed.rst

.. include:: generated/failed.rst

.. include:: generated/skipped.rst

.. include:: generated/testspec.rst

.. _sec-benchmarks:

.. include:: static/benchmarks.rst
