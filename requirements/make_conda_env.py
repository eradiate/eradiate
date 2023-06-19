import os
import sys
from copy import deepcopy
import networkx as nx

import click
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq as CS


@click.command()
@click.option(
    "-s",
    "--sections",
    default="dependencies,main,recommended,tests,dev,docs,optional",
    help="Dependency sections to include in the produced environment.yml file. "
    "Default: 'dependencies,main,recommended,tests,dev,docs,optional'",
)
@click.option("-o", "--output-dir", default="requirements/conda", help="Path to output file.")
@click.option(
    "-l",
    "--layered-config",
    default="./requirements/layered.yml",
    help="Path to layered requirement dependency configuration file. "
    "Default: ./requirements/layered.yml",
)
@click.option(
    "-p",
    "--pip-deps",
    is_flag=True,
    default=False,
    help="Include pip dependencies in the environment.yml file. Default: False",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress terminal output.")
@click.option(
    "-g",
    "--write-graphs",
    is_flag=True,
    default=False,
    help="Write dependency graphs to Graphviz dot files. Default: False"
)
def cli(sections, output_dir, layered_config, pip_deps, quiet, write_graphs):
    """
    Create a Conda environment file from a pyproject.toml.
    """

    # Set YAML parameters
    yaml = YAML(typ="rt")  # Round-trip mode allows for comment insertion
    indent_offset = 2
    yaml.indent(offset=indent_offset)

    # Load environment file template
    with open(os.path.join("requirements", "environment.in")) as f:
        env_yml = yaml.load(f.read())
    with open(layered_config) as f:
        layered_yml = yaml.load(f.read())

    sections = [
        (x.strip(), {"packages": set(layered_yml[x.strip()].get("packages", []))})
        for x in sections.split(",")
    ]
    dep_list = deepcopy(env_yml["dependencies"]) if "dependencies" in env_yml else []

    # Create dependency graph
    G = nx.DiGraph()
    G.add_nodes_from(sections)
    for section, data in sections:
        for include in layered_yml[section].get("includes", []):
            G.add_edge(section, include)

    if write_graphs:
      path = os.path.join(output_dir, f"layer_graph.dot")
      nx.drawing.nx_pydot.write_dot(G, path)

    G.add_node("default", packages=set(dep_list))
    G.add_edge("main", "default")

    if not nx.algorithms.dag.is_directed_acyclic_graph(G):
        raise ValueError("Dependency graph is not a DAG")

    sections = list(nx.algorithms.dag.topological_sort(G))[::-1]

    for section in sections:
        # Select sections to include
        subG = G.subgraph(nx.algorithms.dag.descendants(G, section) | {section})
        environment_packages = dict(sorted(subG.nodes.data()))

        # Create dependency list
        section_indices = dict()
        added_packages = set()
        ordered_packages = []
        for included_section, data in environment_packages.items():
            packages_to_add = sorted(data["packages"] - added_packages)

            # Temporary fix until an eradiate_mitsuba conda package is made available
            # This is disabled by default because the Poetry solver fails to find the
            # wheel for DrJIT on OSX.
            mitsuba_pkg = set(p for p in packages_to_add if "eradiate-mitsuba" in p)
            if mitsuba_pkg:
                packages_to_add = [p for p in packages_to_add if p not in mitsuba_pkg]

            section_indices[included_section] = len(ordered_packages)
            if packages_to_add:
                ordered_packages.extend(packages_to_add)

            # Temporary fix until an eradiate_mitsuba conda package is made available
            # This is disabled by default because the Poetry solver fails to find the
            # wheel for DrJIT on OSX.
            if mitsuba_pkg and pip_deps:
                if "pip" not in added_packages:
                    ordered_packages.append("pip")
                ordered_packages.append({"pip": list(mitsuba_pkg)})

            added_packages |= data["packages"]

        if section == "default":
            continue

        # Format dependency list
        lst = CS(ordered_packages)
        for included_section, i in section_indices.items():
            lst.yaml_set_comment_before_after_key(i, included_section, indent_offset)
        env_yml["dependencies"] = lst

        # Output to terminal
        if not quiet:
            yaml.dump(env_yml, sys.stdout)

        # Output to file
        if output_dir is not None:
            path = os.path.join(output_dir, f"environment-{section}.yml")
            with open(path, "w") as outfile:
                if not quiet:
                    print()
                print(f"Saving to {path}")
                yaml.dump(env_yml, outfile)

        if write_graphs:
            path = os.path.join(output_dir, f"environment-{section}.dot")
            if "default" in subG:
                subG = subG.copy()
                subG.remove_node("default")
            nx.drawing.nx_pydot.write_dot(subG, path)

if __name__ == "__main__":
    cli()
