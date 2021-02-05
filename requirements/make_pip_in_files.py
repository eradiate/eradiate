import os

import click
from setuptools.config import read_configuration
from ruamel.yaml import YAML


@click.command()
@click.option(
    "-s",
    "--sections",
    default="main,tests,dev,docs",
    help="Dependency sections to include in the produced environment.yml file. "
    "Default: 'main,tests,dev,docs'",
)
@click.option(
    "-i",
    "--input",
    default="setup.cfg",
    help="Path to setup.cfg file. Default: setup.cfg",
)
@click.option(
    "-o",
    "--output-dir",
    default="./requirements",
    help="Path to output directory. Default: ./requirements",
)
@click.option(
    "-l",
    "--layered-config",
    default="./requirements/layered.yml",
    help="Path to layered requirement dependency configuration file. "
    "Default: ./requirements/layered.yml",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress terminal output.")
def cli(sections, input, output_dir, layered_config, quiet):
    # Load dependency list
    if not quiet:
        print(f"Reading dependencies from {input}")

    setup_config = read_configuration(input)
    sections = [x.strip() for x in sections.split(",")]

    # Load layered dependency dependencies
    yaml = YAML(typ="safe")
    with open(layered_config) as f:
        layered_yml = yaml.load(f.read())

    for section in sections:
        if not quiet:
            print(f"Processing section '{section}'")

        if section == "main":
            packages = setup_config["options"]["install_requires"]
        else:
            packages = setup_config["options"]["extras_require"][section]

        if not quiet:
            print(f"Writing to {os.path.join(output_dir, f'{section}.in')}")

        # Create .in file
        with open(os.path.join(output_dir, f"{section}.in"), "w") as f:
            # Prepend layered requirement includes
            includes = layered_yml[section].get("includes", [])
            for include in includes:
                f.write(f"-r {include}.in\n")

            constraints = layered_yml[section].get("constraints", [])
            for constraint in constraints:
                f.write(f"-c {constraint}.txt\n")

            f.write("\n".join(packages))
            f.write("\n")


if __name__ == "__main__":
    cli()
