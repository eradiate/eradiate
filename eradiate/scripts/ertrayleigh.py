import click
import yaml
import matplotlib.pyplot as plt

from eradiate.solvers.onedim.rayleigh import RayleighSolverApp


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-q", "--quiet", is_flag=True, help="Hide progress information.")
def cli(config, output, quiet):
    """A simple command-line interface to the RayleighSolverApp class.

    This tool reads a RayleighSolverApp YAML configuration file located at
    CONFIG and simulates radiative transfer on it. It then plots results
    to a file located at OUTPUT.
    """

    # Load configuration
    with open(config, "r") as configfile:
        config = yaml.safe_load(configfile)
    app = RayleighSolverApp(config)

    # Run simulation
    app.compute(quiet=quiet)

    # Post-process and plot results
    app.plot()
    # TODO: transfer that to RayleighSolverApp
    print(f"Saving to {output} ...")
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    cli()
