import click
import ruamel.yaml as yaml

from eradiate.solvers.rami import RamiSolverApp


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("fname_results", type=click.Path())
@click.argument("fname_plots", type=click.Path())
def cli(config, fname_results, fname_plots):
    """A simple command-line interface to the RamiSolverApp class.

    This tool reads a RamiSolverApp YAML configuration file located at
    CONFIG and simulates radiative transfer on it. It stores the results in
    files named with the prefix FNAME_RESULTS and creates default plots in
    files named with the prefix FNAME_PLOTS.
    """

    # Load configuration
    with open(config, "r") as configfile:
        config = yaml.safe_load(configfile)
    app = RamiSolverApp.from_dict(config)

    # Run simulation
    app.run()

    # Save and plot results
    app.save_results(fname_results)
    app.plot_results(fname_plots)


if __name__ == "__main__":
    cli()
