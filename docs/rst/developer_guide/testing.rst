.. _sec-developer_guide-testing:

Testing and test report
=======================

Eradiate is shipped with a series of tests written with `pytest <https://docs.pytest.org/en/latest/>`_.
Tests are grouped by their scope and folders are named to enforce a fixed order of execution for the tests.

At the highest level, there is a separation of tests for mitsuba plugins which are maintained in the Eradiate codebase and tests for Eradiate code itself.
The tests for Eradiate are then grouped by their complexity. First unit tests are executed, followed by system tests and finally regression tests.

Running the tests
-----------------

To run the test suite, invoke ``pytest`` with the following command:

.. code-block:: bash

    pytest tests

The Mitsuba test suite can also be run:

.. code-block:: bash

    pytest ext/mitsuba2/src

Testing guidelines
------------------

Writing test specification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Eradiate's tests can be roughly categorised as follows:

- unit tests focus on the smallest testable units of code;
- system tests check the behaviour of entire applications;
- regression tests which compare simulation results with previous versions.

While categorising each individual test is not always an easy task, this nomenclature highlights the fact that tests have varied degrees of complexity. When the rationale, setup and course of action of a test is not obvious by reading the corresponding source code, properly documenting it in a structured way is crucial. For this reason, Eradiate defines a test description template to be used for system and regression tests. Documented tests have a dedicated section in the test report.

The test specification consists of three main parts: the **description of the test rationale**, the **details of the setup**, explaining in prose, how a test is designed and, finally, the **expected outcome** of the test, which describes based on what the test should pass or fail. 

The following template can be copied to new test cases and the information filled in as needed. Note that we strongly suggest using literal strings (prefixed with a ``r``) in order to avoid issues with escape sequences.

.. code-block:: none

    r"""
    Test title
    ==========
    
    :Description: This is the short description of the test case
    
    Rationale
    ---------

        This is some explanatory text

        - This section explaines the details
        - Of how the test is implemented
        - It can contain math! :math:`e^{i\pi}=-1`

    Expected behaviour
    ------------------

        This section explaines the expected result of the test and how it is asserted.

        - We assert that something was calculated
        - Additionally the result must be correct
        
    """

The test specification can hold any valid restructured text. A quick rundown on that can be found
`here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ .

Upon report generation, the test result and location of the test file will
be added and the result will look similar to this:

.. image:: ../../fig/spec_render.png
   :scale: 50 %

The test specification of unit tests is not parsed for the test report and does not have to comply with these guidelines. For those a short explanation is sufficient, but the three general parts mentioned above should still serve as a guideline for relevant and helpful test specification.

Regression tests
----------------

Eradiate's regression tests are designed to allow the monitoring of results over time. Each test produces a NetCDF file with the current results as well as an image containing plots and metrics, comparing the
current version of Eradiate to the reference results. The results of these tests can be archived for future reference.

To run the regression tests isolated from the rest of the test suite, we introduced the ``regression`` fixture. To run only the regression tests, invoke pytest like this:

.. code-block:: bash

    pytest tests -m "regression" --artefact_dir <a directory of your choice>

The ``artefact_dir`` parameter defines the output directory in which the results and plots will be placed. If the directory does not exist, it will be created. The artefact directory defaults to ``./build/test_artefacts``, which is resolved relative to the current working directory.

Test report
-----------

The Eradiate test report facilities have been moved to a separate repository, which can be found on `github <https://github.com/eradiate/eradiate-test-report>`_ .