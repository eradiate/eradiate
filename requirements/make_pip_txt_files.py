import os

import click
from ruamel.yaml import YAML


@click.command()
@click.option(
    "-s",
    "--sections",
    default="dependencies,recommended",
    help="Dependency sections to include in the produced environment.yml file. "
    "Default: 'dependencies,recommended'",
)
@click.option(
    "-o",
    "--output-dir",
    default="./build",
    help="Path to output directory. Default: ./build",
)
@click.option(
    "-l",
    "--layered-config",
    default="./requirements/layered.yml",
    help="Path to layered requirement dependency configuration file. "
    "Default: ./requirements/layered.yml",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress terminal output.")
def cli(sections, output_dir, layered_config, quiet):
    # Load dependency list

    sections = [x.strip() for x in sections.split(",")]

    # Load layered dependency dependencies
    yaml = YAML(typ="safe")
    with open(layered_config) as f:
        layered_yml = yaml.load(f.read())

    for section in sections:
        if not quiet:
            print(f"Processing section '{section}'")

        packages = layered_yml[section].get("packages", [])
        includes = layered_yml[section].get("includes", [])
        for include in includes:
            packages += layered_yml[include].get("packages", [])
        constraints = layered_yml[section].get("constraints", [])
        for constraint in constraints:
            packages += layered_yml[constraint].get("packages", [])

        if not quiet:
            print(f"Writing to {os.path.join(output_dir, f'{section}.txt')}")

        # Create .txt file
        with open(os.path.join(output_dir, f"{section}.txt"), "w") as f:

            f.write("\n".join(packages))
            f.write("\n")


if __name__ == "__main__":
    cli()
