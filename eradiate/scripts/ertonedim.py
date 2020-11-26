import click
import ruamel.yaml as yaml

from eradiate.solvers.onedim.app import OneDimSolverApp


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("fname_results", type=click.Path())
@click.argument("fname_plots", type=click.Path())
# @click.option("-q", "--quiet", is_flag=True, help="Hide progress information.")
def cli(config, fname_results, fname_plots):
    """A simple command-line interface to the OneDimSolverApp class.

    This tool reads a OneDimSolverApp YAML configuration file located at
    CONFIG and simulates radiative transfer on it. It stores the results in
    files named with the prefix FNAME_RESULTS and creates default plots in
    files named with the prefix FNAME_PLOTS.
    """

    # Load configuration
    with open(config, "r") as configfile:
        config = yaml.safe_load(configfile)
    app = OneDimSolverApp(config)

    # Run simulation
    app.run()

    # Save and plot results
    app.save_results(fname_results)
    app.plot_results(fname_plots)


if __name__ == "__main__":
    cli()
