Testing and test report
=======================

Running the tests
-----------------

To run the test suite, invoke ``pytest`` with the following command:

.. code-block:: bash

    pytest eradiate

The Mitsuba test suite can also be run:

.. code-block:: bash

    pytest ext/mitsuba2/src
    
Test report
-----------

Eradiate can generate a html based test report, collecting information about
the number of tests, their outcome and which will collect the test specification
for more complex test cases, such as integration and system tests.

.. code-block:: bash

   python test_report/generate_report.py

The resulting report will be located in :code:`$ERADIATE_DIR/build/html_test-report`

Testing guidelines
------------------

Writing test specification
^^^^^^^^^^^^^^^^^^^^^^^^^^

While unit tests focus on the smallest testable units of code in Eradiate, integration
and system tests emphasize the interaction of components or even the entire software.

Since the test idea and setup may not be readily understandable from looking at the source code
of these tests, they are documented in a structured way and their specification is added in a separate section in the test report.

The test specification consists of three main parts: The **description of the test idea**,
the **details of the setup**, explaining in prose, how a test is designed and finally the **expected outcome** of the test, detailing what is checked.
The following template can be copied to new test cases and the information filled in as needed.

.. code-block:: none

    r"""
    Test title
    ----------
    
    :Description: This is the short description of the test case
    
    Rationale
    ^^^^^^^^^

        This is some explanatory text

        - This section explaines the details
        - Of how the test is implemented
        - It can contain math! :math:`e^{i\pi}=-1`

    Expected behaviour
    ^^^^^^^^^^^^^^^^^^

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