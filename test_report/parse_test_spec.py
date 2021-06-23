import importlib.util
import inspect
import json
import os
import pathlib
import re


def get_files(path):
    """
    Returns all files in 'path' whose name starts with 'test_' and ends with '.py'
    as a list
    """
    testfiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find("test_") == 0 and os.path.splitext(file)[1] == ".py":
                testfiles.append(os.path.join(root, file))
    return testfiles


def parse_tests(file):
    """
    Extracts the docstrings for all classes and creates a dictionary mapping
    test case names to their doc strings.
    """
    module_name = os.path.splitext(os.path.basename(file))[0]
    spec = importlib.util.spec_from_file_location(module_name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    returndict = dict()
    for name, value in inspect.getmembers(mod):
        if inspect.isfunction(value):
            if name.find("test_") == 0:
                returndict[name] = value.__doc__
    return returndict


def remove_indentation(doc):
    """
    Removes the indentation from a docstring.
    The number of spaces in the *second* line of a docstring is considered
    the indentation level and that amount of spaces well be removed from
    the beginning of each line
    """
    lines = doc.split("\n")
    regex = re.compile("^[ ]+")
    spaces = regex.match(lines[1]).group()

    return "\n".join([line.replace(spaces, "", 1) for line in lines])


def get_testcase_outcome(report, name):
    """
    Returns the outcome of the testcase 'name' from 'report'.
    If no result is found, return None.
    """
    returndict = dict()
    for test in report["tests"]:
        testname = test["nodeid"].split("::")[-1]

        if testname.find(name) != -1:
            if testname.find("[") != -1:
                # split off the variant, which is enclosed by [ ]
                variant = testname.split("[")[-1][:-1]
            else:
                variant = "no variant"
            returndict[variant] = test["outcome"]

    # if no variant is found, name should only occur once in the test report
    if "no variant" in returndict and len(returndict) > 1:
        raise KeyError(
            f"Inconsistent parametrizations found for test {name}. Testcase name clash?"
        )
    if len(returndict) == 0:
        return {"no variant": "Result not found!"}
    else:
        return returndict


def get_testcase_location(report, name):
    """
    Return the file location for a testcase
    """
    for test in report["tests"]:
        testlocation, testname = test["nodeid"].split("::")
        if testname.find(name) != -1:
            return testlocation
    else:
        return "Test case not found!"


def get_testcase_metric(report, name):
    """
    Get the testcase metric if it exists and concatenate
    """
    metrics = None

    try:
        for test in report["tests"]:
            testname = test["nodeid"].split("::")[-1]
            if testname.find(name) != -1:
                metrics = test["metadata"]["metrics"]
    except KeyError:
        return ""

    if metrics is None:
        return ""

    returnstring = "\nMetrics\n-------\n\n"
    for id, values in metrics.items():
        returnstring += (
            values["name"]
            + "\n"
            + '"' * len(values["name"])
            + "\n\n"
            + values["description"]
            + " "
            + values["value"]
            + " "
            + values["units"]
            + "\n\n"
        )

    return returnstring


def update_docstring(doc, name, report):
    """
    Split the heading off the docstring and insert a line containing
    the result of the respective test case.
    """

    outcomes = get_testcase_outcome(report, name)
    location = get_testcase_location(report, name)
    metric = get_testcase_metric(report, name)

    doc = remove_indentation(doc)
    splitstring = "\n\nRationale\n---------\n\n"
    [firstpart, lastpart] = doc.split(splitstring)

    result = ""
    if len(outcomes) == 1:
        outcome = outcomes["no variant"]
        if outcome == "passed":
            result = f":green:`{outcome}`"
        elif outcome == "failed":
            result = f" :red:`{outcome}`"
        else:
            result = outcome
    else:
        for variant, outcome in outcomes.items():
            if outcome == "passed":
                outcomes[variant] = f":green:`{outcome}`"
            elif outcome == "failed":
                outcomes[variant] = f":red:`{outcome}`"
        result = "\n* ".join(
            [f"{variant}: {outcomes[variant]}" for variant in sorted(outcomes)]
        )

    resultstring = f"""
:Test result: 

* {result}

"""
    locationstring = f":Test location: {location}"

    firstpart = "\n".join([firstpart, resultstring, locationstring])
    doc = "".join([firstpart, splitstring, lastpart, metric, "\n\n--------\n\n"])

    return doc


def write_to_file(test_dict, output_dir):
    """
    Write the rst file containing the updated documentation for all system tests.
    """

    header = """
.. role:: red

.. role:: green

.. _sec-testspec:

*********************************
Detailed system-test results
*********************************

This section contains the specification for the system and system tests in Eradiate.
The tests are documented by presenting their general concept, e.g. what is tested, followed
by the test setup and execution, which includes critical points about the implementation and
finally the expected behaviour of the software under test and how the success or failure of
the test is asserted.    
"""

    body = ""
    for _, doc in test_dict.items():
        body += doc
        body += "\n"

    with open(output_dir / "testspec.rst", "w") as specfile:
        specfile.write(header)
        specfile.write(body)


def generate():
    print("Parsing test specification and generating documents for report")
    eradiate_dir = pathlib.Path(os.environ["ERADIATE_DIR"])
    test_dir = eradiate_dir / "eradiate" / "tests" / "system"
    output_dir = eradiate_dir / "test_report" / "generated"
    if not pathlib.Path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_dir / "report_eradiate.json") as json_data:
        report = json.load(json_data)

    testfiles = get_files(test_dir)

    test_dict = dict()
    for file in testfiles:
        test_dict.update(parse_tests(file))

    for name, doc in test_dict.items():
        doc = update_docstring(doc, name, report)
        test_dict.update({name: doc})

    write_to_file(test_dict, output_dir)


if __name__ == "__main__":
    generate()
