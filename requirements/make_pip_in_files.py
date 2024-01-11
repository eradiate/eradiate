import os

import click
from _utils import resolve_package_manager
from ruamel.yaml import YAML


@click.command()
@click.option(
    "-s",
    "--sections",
    default="dependencies,main,recommended,tests,dev,docs,optional",
    help="Dependency sections to include in the produced environment.yml file. "
    "Default: 'dependencies,main,recommended,tests,dev,docs,optional'",
)
@click.option(
    "-o",
    "--output-dir",
    default="./requirements/pip",
    help="Path to output directory. Default: ./requirements/pip",
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
        layered_yml = resolve_package_manager(yaml.load(f.read()), "pip")

    for section in sections:
        if not quiet:
            print(f"Processing section '{section}'")

        packages = layered_yml[section].get("packages", [])

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
                f.write(f"-c {constraint}.lock.txt\n")

            f.write("\n".join(packages))
            f.write("\n")


if __name__ == "__main__":
    cli()
