import pytest
import os
import ruamel.yaml as yaml

ROOTDIR = os.getenv("ERADIATE_DIR")
DEPDIR = os.path.join(ROOTDIR, "resources", "deps")
OPERATORS = ["==", "~=", ">=", "!=", "="]


@pytest.mark.parametrize("condafile, pipfile", [("requirements_conda.yml", "requirements_pip.txt"),
                                                ("requirements_dev_conda.yml", "requirements_dev_pip.txt")])
def test_file_synchronicity(condafile, pipfile):
    """Test the synchronicity of the conda and pip requirements files, by comparing
    contents:

    All packages in the conda yaml file must be present in the pip txt file.
    Also all specified versions must match.
    """

    with open(os.path.join(DEPDIR, condafile), "r") as yamlfile:
        yamldata = yaml.load(yamlfile, Loader=yaml.SafeLoader)

    yamldict = dict()
    for item in yamldata["dependencies"]:
        if isinstance(item, dict):
            continue
        print(item)
        for op in OPERATORS:
            if op in item:
                key, value = item.split(op)
                if key == "pip" or key =="python":
                    continue
                yamldict[key] = value
                break

    with open(os.path.join(DEPDIR, pipfile), "r") as txtfile:
        txtdata = txtfile.read().strip().splitlines()

    txtdict = dict()
    for item in txtdata:
        for op in OPERATORS:
            if op in item:
                key, value = item.split(op)
                txtdict[key] = value
                break

    for key, value in yamldict.items():
        assert key in txtdict, f"{key} is present in conda deps but missing in pip deps."
        assert yamldict[key] == txtdict[key], f"Versions not matching for '{key}': conda version is " \
                                              f"{yamldict[key]}, pip version is {txtdict[key]}"